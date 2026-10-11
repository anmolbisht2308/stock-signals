import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from stock_signals import job_runner
from stock_signals.api import admin_token, create_api
from stock_signals.config import Settings
from stock_signals.job_runner import JobError, JobRunner, build_argv, default_command, redactor
from stock_signals.jobs import App

# Child processes stand in for the CLI: each job kind maps to a tiny Python script.
SCRIPTS = {
    "update": "print('downloading'); print('bot123456:ABCDEFGHIJKLMNOPQRSTUVWXYZ0123 ok')",
    "scan": "import sys; print('boom'); sys.exit(3)",
    "evening": "import time\nfor i in range(400):\n    print(i, flush=True); time.sleep(0.01)",
    "telegram-test": "print('hello')",
    "review": "\n".join(f"print({i})" for i in range(30)),
}


def fake_command(argv: list[str]) -> list[str]:
    return [sys.executable, "-c", SCRIPTS[argv[0]]]


@pytest.fixture
def runner(db):
    r = JobRunner(db, redactor(["SECRET-PASSWORD"]), command=fake_command)
    yield r
    for proc in list(r.procs.values()):
        proc.kill()


def run(r: JobRunner, kind: str, params: dict | None = None) -> dict:
    job = r.start(kind, params)
    r.wait(job["id"], timeout=20)
    return r.get(job["id"])


def wait_for(cond, timeout: float = 10) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def test_build_argv_maps_params_to_cli_flags():
    assert build_argv("evening", {"no_update": True})[2] == ["evening", "--no-update"]
    assert build_argv("backtest", {"book": "swing", "start": "2022-01-01"})[2] == [
        "backtest",
        "--book",
        "swing",
        "--start",
        "2022-01-01",
    ]
    assert build_argv("walkforward", {"final": True})[2][-1] == "--final"
    assert build_argv("scan", {})[2] == ["scan"]
    assert default_command(["update"])[1:] == ["-m", "stock_signals.cli", "update"]


@pytest.mark.parametrize(
    ("kind", "params", "status"),
    [
        ("rm-rf", {}, 404),
        ("serve", {}, 404),  # long-running commands are not jobs
        ("backtest", {"start": "2016/01/01"}, 422),
        ("backtest", {"book": "intraday"}, 422),
        ("update", {"symbols": "; rm -rf /"}, 422),  # unknown params are rejected
    ],
)
def test_build_argv_rejects_bad_input(kind, params, status):
    with pytest.raises(JobError) as exc:
        build_argv(kind, params)
    assert exc.value.status == status


def test_redactor_hides_secrets():
    redact = redactor(["s3cret-token", None, ""])
    assert redact("token s3cret-token here") == "token <redacted> here"
    assert "ABCDEF" not in redact("POST https://api.telegram.org/bot123456:ABCDEFGHIJKLMNOPQRST/x")
    assert redact("mongodb+srv://me:pw@cluster0/db") == "mongodb+srv://<redacted>@cluster0/db"


def test_job_runs_and_streams_redacted_output(runner):
    job = run(runner, "update")
    assert job["status"] == "succeeded" and job["exit_code"] == 0
    assert job["command"] == "stock-signals update"
    assert job["log"][0] == "downloading"
    assert "ABCDEFG" not in job["log"][1] and "<redacted>" in job["log"][1]
    assert runner.locks.count_documents({}) == 0


def test_failed_job_records_exit_code(runner):
    job = run(runner, "scan")
    assert job["status"] == "failed" and job["exit_code"] == 3
    assert job["error"] == "Exit code 3" and job["log"] == ["boom"]


def test_one_heavy_job_at_a_time(runner):
    long = runner.start("evening")
    with pytest.raises(JobError) as exc:
        runner.start("update")
    assert exc.value.status == 409 and "evening" in str(exc.value)
    assert run(runner, "telegram-test")["status"] == "succeeded"  # light jobs still run
    runner.wait(long["id"], timeout=20)
    assert runner.get(long["id"])["status"] == "succeeded"
    assert run(runner, "update")["status"] == "succeeded"  # lock was released


def test_parallel_starts_get_exactly_one_heavy_job(runner):
    results: list[str] = []
    barrier = threading.Barrier(8)

    def go():
        barrier.wait()
        try:
            runner.start("evening")
            results.append("ok")
        except JobError as exc:
            results.append(str(exc.status))

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["409"] * 7 + ["ok"]
    assert runner.jobs.count_documents({"kind": "evening"}) == 1


def test_cancel_stops_the_process_and_frees_the_lock(runner):
    job = runner.start("evening")
    wait_for(lambda: runner.get(job["id"])["log_total"] > 0)
    runner.cancel(job["id"])
    runner.wait(job["id"], timeout=20)
    done = runner.get(job["id"])
    assert done["status"] == "cancelled" and done["error"] is None
    assert "⏹️ Cancel requested" in done["log"]
    assert runner.locks.count_documents({}) == 0
    with pytest.raises(JobError) as exc:
        runner.cancel(job["id"])
    assert exc.value.status == 409


def test_log_since_survives_trimming(runner, monkeypatch):
    monkeypatch.setattr(job_runner, "MAX_LOG_LINES", 10)
    job = run(runner, "review")  # prints 0..29
    assert job["log_total"] == 30 and job["log"] == [str(i) for i in range(20, 30)]
    assert runner.get(job["id"], since=25)["log"] == ["25", "26", "27", "28", "29"]
    assert runner.get(job["id"], since=30)["log"] == []


def test_recover_marks_orphaned_jobs(runner):
    job = runner.start("evening")
    runner.procs[job["id"]].kill()
    runner.wait(job["id"], timeout=20)
    # Simulate an API crash mid-job: the doc still says running and the lock is held.
    runner.jobs.update_one({"_id": job["id"]}, {"$set": {"status": "running"}})
    runner.locks.insert_one({"_id": "heavy", "job_id": job["id"]})
    fresh = JobRunner(runner.jobs.database, command=fake_command)
    assert fresh.recover() == 1
    assert fresh.get(job["id"])["status"] == "failed"
    assert fresh.locks.count_documents({}) == 0


def test_real_cli_telegram_test_job(db, tmp_path):
    """The default command really runs the CLI (no Telegram token → it just logs)."""
    r = JobRunner(db)
    job = run(r, "telegram-test")
    assert job["status"] == "succeeded", job["log"]
    assert any("Hello from stock-signals" in line for line in job["log"])


@pytest.fixture
def client(db, tmp_path, runner):
    settings = Settings(data_dir=tmp_path / "data", admin_password="pw")
    c = TestClient(create_api(App.create(settings, db=db), runner=runner))
    c.headers["Authorization"] = f"Bearer {admin_token('pw')}"
    return c


def test_jobs_api(client, runner):
    anon = TestClient(client.app)
    assert anon.post("/api/jobs", json={"kind": "update"}).status_code == 401

    res = client.post("/api/jobs", json={"kind": "update"})
    assert res.status_code == 201
    job_id = res.json()["id"]
    runner.wait(job_id, timeout=20)
    job = client.get(f"/api/jobs/{job_id}?since=1").json()
    assert job["status"] == "succeeded" and job["log_total"] == 2 and len(job["log"]) == 1

    listing = client.get("/api/jobs").json()
    assert listing["jobs"][0]["id"] == job_id and "log" not in listing["jobs"][0]
    assert listing["running"] == []

    assert client.post("/api/jobs", json={"kind": "nope"}).status_code == 404
    bad = client.post("/api/jobs", json={"kind": "scan", "params": {"date": "x"}})
    assert bad.status_code == 422 and "date" in bad.json()["detail"]
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409

    long = client.post("/api/jobs", json={"kind": "evening"}).json()
    busy = client.post("/api/jobs", json={"kind": "update"})
    assert busy.status_code == 409
    assert client.get("/api/jobs").json()["running"][0]["id"] == long["id"]
    assert client.post(f"/api/jobs/{long['id']}/cancel").status_code == 200
    runner.wait(long["id"], timeout=20)


def test_pause_and_resume(client):
    assert client.post("/api/pause").json() == {"paused": True}
    assert client.get("/api/dashboard").json()["paused"] is True
    assert client.post("/api/resume").json() == {"paused": False}
    assert client.get("/api/dashboard").json()["paused"] is False
