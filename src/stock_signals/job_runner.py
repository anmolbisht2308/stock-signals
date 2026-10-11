"""Background jobs for the admin panel (Phase 10.3).

Each panel button runs the same CLI command you would type (`stock-signals evening` …) in a
child process. Output lines are stored on the job document in the `jobs` collection as they
arrive, so the panel can show a live log. Only one heavy job (anything that loads the price
panel or downloads data) runs at a time; the lock is a document in `job_locks`, so a second
request fails fast instead of queueing.

The CLI sends its result to Telegram as usual, so a job started from the panel behaves exactly
like one started from cron.
"""

import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from pymongo import DESCENDING
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

log = logging.getLogger(__name__)

MAX_LOG_LINES = 5000
DATE = r"^\d{4}-\d{2}-\d{2}$"
Book = Literal["swing", "positional", "all"]


class NoParams(BaseModel, extra="forbid"):
    pass


class EveningParams(BaseModel, extra="forbid"):
    no_update: bool = False


class ScanParams(BaseModel, extra="forbid"):
    date: str | None = Field(default=None, pattern=DATE)


class BacktestParams(BaseModel, extra="forbid"):
    book: Book = "positional"
    start: str = Field(default="2016-01-01", pattern=DATE)
    end: str | None = Field(default=None, pattern=DATE)


class WalkforwardParams(BaseModel, extra="forbid"):
    book: Book = "positional"
    final: bool = False


class BasketBacktestParams(BaseModel, extra="forbid"):
    start: str = Field(default="2016-01-01", pattern=DATE)
    end: str | None = Field(default=None, pattern=DATE)


@dataclass(frozen=True)
class JobSpec:
    params: type[BaseModel]
    heavy: bool
    argv: Callable[[BaseModel], list[str]]


def _opt(flag: str, value: str | None) -> list[str]:
    return [flag, value] if value else []


SPECS: dict[str, JobSpec] = {
    "update": JobSpec(NoParams, True, lambda p: ["update"]),
    "evening": JobSpec(
        EveningParams, True, lambda p: ["evening", *(["--no-update"] if p.no_update else [])]
    ),
    "scan": JobSpec(ScanParams, True, lambda p: ["scan", *_opt("--date", p.date)]),
    "backtest": JobSpec(
        BacktestParams,
        True,
        lambda p: ["backtest", "--book", p.book, "--start", p.start, *_opt("--end", p.end)],
    ),
    "walkforward": JobSpec(
        WalkforwardParams,
        True,
        lambda p: ["walkforward", "--book", p.book, *(["--final"] if p.final else [])],
    ),
    "basket-backtest": JobSpec(
        BasketBacktestParams,
        True,
        lambda p: ["basket-backtest", "--start", p.start, *_opt("--end", p.end)],
    ),
    "review": JobSpec(NoParams, True, lambda p: ["review"]),
    "telegram-test": JobSpec(NoParams, False, lambda p: ["telegram-test"]),
}


class JobError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def build_argv(kind: str, params: dict | None) -> tuple[JobSpec, BaseModel, list[str]]:
    spec = SPECS.get(kind)
    if spec is None:
        raise JobError(404, f"Unknown job: {kind}")
    try:
        parsed = spec.params.model_validate(params or {})
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(x) for x in first["loc"]) or "params"
        raise JobError(422, f"{field}: {first['msg']}") from exc
    return spec, parsed, spec.argv(parsed)


_SECRET_PATTERNS = [
    (re.compile(r"bot\d+:[A-Za-z0-9_-]{20,}"), "bot<redacted>"),
    (re.compile(r"(mongodb(?:\+srv)?://)[^@/\s]+@"), r"\1<redacted>@"),
]


def redactor(secrets: list[str | None]) -> Callable[[str], str]:
    """Strip the bot token and Mongo credentials from log lines before they are stored."""
    exact = [s for s in secrets if s and len(s) >= 6]

    def redact(line: str) -> str:
        for s in exact:
            line = line.replace(s, "<redacted>")
        for pattern, repl in _SECRET_PATTERNS:
            line = pattern.sub(repl, line)
        return line

    return redact


def default_command(argv: list[str]) -> list[str]:
    return [sys.executable, "-m", "stock_signals.cli", *argv]


class JobRunner:
    def __init__(
        self,
        db: Database,
        redact: Callable[[str], str] = lambda s: s,
        command: Callable[[list[str]], list[str]] = default_command,
    ) -> None:
        self.jobs = db["jobs"]
        self.locks = db["job_locks"]
        self.redact = redact
        self.command = command
        self.procs: dict[str, subprocess.Popen] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.jobs.create_index([("created_at", DESCENDING)])

    def recover(self) -> int:
        """Jobs left running by a stopped API can't finish any more: mark them and free the lock."""
        now = datetime.now(UTC)
        res = self.jobs.update_many(
            {"status": {"$in": ["queued", "running", "cancelling"]}},
            {
                "$set": {"status": "failed", "finished_at": now, "error": "API restarted"},
                "$push": {"log": "⚠️ The API stopped while this job was running."},
                "$inc": {"log_seq": 1},
            },
        )
        self.locks.delete_many({})
        return res.modified_count

    def start(self, kind: str, params: dict | None = None) -> dict:
        spec, parsed, argv = build_argv(kind, params)
        job_id = uuid.uuid4().hex
        if spec.heavy:
            try:
                self.locks.insert_one({"_id": "heavy", "job_id": job_id})
            except DuplicateKeyError:
                busy = self.locks.find_one({"_id": "heavy"}) or {}
                running = self.jobs.find_one({"_id": busy.get("job_id")}, {"kind": 1}) or {}
                name = running.get("kind", "another job")
                raise JobError(409, f"{name} is still running. Wait for it or cancel it.") from None
        doc = {
            "_id": job_id,
            "kind": kind,
            "params": parsed.model_dump(),
            "command": "stock-signals " + " ".join(argv),
            "heavy": spec.heavy,
            "status": "running",
            "created_at": datetime.now(UTC),
            "started_at": datetime.now(UTC),
            "finished_at": None,
            "exit_code": None,
            "error": None,
            "log": [],
            "log_seq": 0,
        }
        self.jobs.insert_one(doc)
        try:
            proc = subprocess.Popen(
                self.command(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as exc:
            self._finish(job_id, spec.heavy, "failed", None, f"Could not start: {exc}")
            raise JobError(500, f"Could not start {kind}: {exc}") from exc
        self.procs[job_id] = proc
        thread = threading.Thread(target=self._pump, args=(job_id, proc, spec.heavy), daemon=True)
        self.threads[job_id] = thread
        thread.start()
        return self.get(job_id) or doc

    def _pump(self, job_id: str, proc: subprocess.Popen, heavy: bool) -> None:
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = self.redact(raw.rstrip("\n"))
                self._append(job_id, line)
            code = proc.wait()
        except Exception as exc:  # keep the lock from leaking if anything above breaks
            log.exception("job %s pump failed", job_id)
            self._finish(job_id, heavy, "failed", None, str(exc))
            return
        finally:
            self.procs.pop(job_id, None)
        current = self.jobs.find_one({"_id": job_id}, {"status": 1}) or {}
        if current.get("status") == "cancelling":
            status = "cancelled"
        else:
            status = "succeeded" if code == 0 else "failed"
        self._finish(job_id, heavy, status, code, None if code == 0 else f"Exit code {code}")

    def _append(self, job_id: str, line: str) -> None:
        """Add one log line. `log_seq` counts every line ever written, so readers can ask for
        "lines after N" even once the oldest lines have been trimmed."""
        self.jobs.update_one(
            {"_id": job_id},
            {
                "$push": {"log": {"$each": [line], "$slice": -MAX_LOG_LINES}},
                "$inc": {"log_seq": 1},
            },
        )

    def _finish(
        self, job_id: str, heavy: bool, status: str, code: int | None, error: str | None
    ) -> None:
        self.jobs.update_one(
            {"_id": job_id},
            {
                "$set": {
                    "status": status,
                    "exit_code": code,
                    "error": None if status == "cancelled" else error,
                    "finished_at": datetime.now(UTC),
                }
            },
        )
        if heavy:
            self.locks.delete_one({"_id": "heavy", "job_id": job_id})

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job is None:
            raise JobError(404, "Job not found")
        proc = self.procs.get(job_id)
        if job["status"] != "running" or proc is None:
            raise JobError(409, f"Job is {job['status']}, not running")
        self.jobs.update_one({"_id": job_id}, {"$set": {"status": "cancelling"}})
        self._append(job_id, "⏹️ Cancel requested")
        proc.terminate()
        return self.get(job_id) or job

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        thread = self.threads.get(job_id)
        if thread:
            thread.join(timeout)

    def get(self, job_id: str, since: int = 0) -> dict | None:
        doc = self.jobs.find_one({"_id": job_id})
        if doc is None:
            return None
        lines = doc.pop("log", [])
        total = doc.pop("log_seq", len(lines))
        dropped = total - len(lines)
        return {**_public(doc), "log_total": total, "log": lines[max(0, since - dropped) :]}

    def recent(self, limit: int = 20) -> list[dict]:
        docs = (
            self.jobs.find({}, {"log": 0, "log_seq": 0}).sort("created_at", DESCENDING).limit(limit)
        )
        return [_public(d) for d in docs]

    def running(self) -> list[dict]:
        docs = self.jobs.find(
            {"status": {"$in": ["running", "cancelling"]}}, {"log": 0, "log_seq": 0}
        )
        return [_public(d) for d in docs]


def _public(doc: dict) -> dict:
    doc = dict(doc)
    doc["id"] = doc.pop("_id")
    return doc
