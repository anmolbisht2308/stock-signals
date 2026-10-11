"""Self-contained HTML backtest report: metrics, equity vs Nifty, drawdown, trades."""

import html
import json
from pathlib import Path

import pandas as pd

from stock_signals.backtest import BacktestResult
from stock_signals.jsonsafe import clean
from stock_signals.messages import reason_label, rs
from stock_signals.metrics import LABELS, fmt

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 760, 260, 64, 16, 12, 28
MAX_POINTS = 800

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--text-primary:#0b0b0b;
--text-secondary:#52514e;--grid:#e4e3df;--series-1:#2a78d6;--ref:#52514e;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242423;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--grid:#33332f;--series-1:#3987e5;--ref:#c3c2b7;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242423;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#33332f;--series-1:#3987e5;--ref:#c3c2b7;}
body{margin:0;background:var(--surface-1);}
.viz-root{background:var(--surface-1);color:var(--text-primary);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:820px;margin:0 auto;
padding:24px 16px;}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.muted{color:var(--text-secondary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.tile{background:var(--surface-2);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:20px}
.chart{position:relative}svg{width:100%;height:auto;display:block}
.grid line{stroke:var(--grid)}.axis text{fill:var(--text-secondary);font-size:11px}
.legend{display:flex;gap:16px;font-size:12px;color:var(--text-secondary);margin:4px 0}
.key{display:inline-block;width:16px;height:2px;vertical-align:middle;margin-right:6px}
.tip{position:absolute;pointer-events:none;background:var(--surface-2);color:var(--text-primary);
border-radius:6px;padding:6px 8px;font-size:12px;display:none;white-space:nowrap;
box-shadow:0 2px 8px rgba(0,0,0,.15)}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:4px 6px;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
.scroll{overflow-x:auto}
"""

JS = (
    """
document.querySelectorAll('.chart[data-series]').forEach(function(box){
  var d=JSON.parse(box.dataset.series),svg=box.querySelector('svg'),tip=box.querySelector('.tip'),
  cross=svg.querySelector('.cross');
  svg.addEventListener('pointermove',function(e){
    var r=svg.getBoundingClientRect(),x=(e.clientX-r.left)*%W%/r.width;
    var i=Math.round((x-%L%)/(%W%-%L%-%R%)*(d.x.length-1));
    if(i<0||i>=d.x.length){tip.style.display='none';return;}
    var px=%L%+i*(%W%-%L%-%R%)/Math.max(d.x.length-1,1);
    cross.setAttribute('x1',px);cross.setAttribute('x2',px);cross.style.display='';
    tip.innerHTML='<b>'+d.x[i]+'</b>'+d.lines.map(function(l){return '<br>'+l.name+': '+l.fmt[i]}).join('');
    tip.style.display='block';var left=(px/%W%)*r.width;
    tip.style.left=Math.min(left+12,r.width-tip.offsetWidth)+'px';tip.style.top='8px';
  });
  svg.addEventListener('pointerleave',function(){tip.style.display='none';cross.style.display='none'});
});
""".replace("%W%", str(W))
    .replace("%L%", str(PAD_L))
    .replace("%R%", str(PAD_R))
)


def _downsample(s: pd.Series) -> pd.Series:
    if len(s) <= MAX_POINTS:
        return s
    step = len(s) / MAX_POINTS
    idx = sorted({int(i * step) for i in range(MAX_POINTS)} | {len(s) - 1})
    return s.iloc[idx]


def _path(values: list[float], lo: float, hi: float) -> str:
    n = max(len(values) - 1, 1)
    span = hi - lo or 1.0
    pts = []
    for i, v in enumerate(values):
        x = PAD_L + i * (W - PAD_L - PAD_R) / n
        y = PAD_T + (hi - v) / span * (H - PAD_T - PAD_B)
        pts.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(pts)


def _chart(title: str, x: list[str], lines: list[dict], fmt_axis, zero_floor: bool = False) -> str:
    """lines: [{name, values, color, dash, fmt}] sharing one y-axis."""
    allv = [v for ln in lines for v in ln["values"] if v == v]
    lo, hi = min(allv), max(allv)
    if zero_floor:
        hi = 0.0
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + (0 if zero_floor else pad)
    ticks = [lo + (hi - lo) * k / 4 for k in range(5)]
    grid = "".join(
        f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
        for y in (PAD_T + (hi - t) / (hi - lo) * (H - PAD_T - PAD_B) for t in ticks)
    )
    labels = "".join(
        f'<text x="{PAD_L - 6}" y="{PAD_T + (hi - t) / (hi - lo) * (H - PAD_T - PAD_B) + 4:.1f}"'
        f' text-anchor="end">{html.escape(fmt_axis(t))}</text>'
        for t in ticks
    )

    def x_label(i: int) -> str:
        anchor = "start" if i == 0 else "end" if i == len(x) - 1 else "middle"
        pos = PAD_L + i * (W - PAD_L - PAD_R) / max(len(x) - 1, 1)
        return f'<text x="{pos:.1f}" y="{H - 8}" text-anchor="{anchor}">{x[i][:7]}</text>'

    xl = "".join(x_label(i) for i in sorted({0, len(x) // 2, len(x) - 1}))
    paths = "".join(
        f'<path d="{_path(ln["values"], lo, hi)}" fill="none" stroke="{ln["color"]}"'
        f' stroke-width="2" stroke-linejoin="round"'
        + (' stroke-dasharray="4 3"' if ln.get("dash") else "")
        + "/>"
        for ln in lines
    )
    legend = (
        '<div class="legend">'
        + "".join(
            f'<span><span class="key" style="background:{ln["color"]}"></span>'
            f"{html.escape(ln['name'])}</span>"
            for ln in lines
        )
        + "</div>"
        if len(lines) > 1
        else ""
    )
    data = {"x": x, "lines": [{"name": ln["name"], "fmt": ln["fmt"]} for ln in lines]}
    return (
        f"<h2>{html.escape(title)}</h2>{legend}"
        f"<div class=\"chart\" data-series='{html.escape(json.dumps(data), quote=True)}'>"
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
        f'<g class="grid">{grid}</g><g class="axis">{labels}{xl}</g>{paths}'
        f'<line class="cross" y1="{PAD_T}" y2="{H - PAD_B}" stroke="var(--text-secondary)"'
        f' stroke-width="1" style="display:none"/></svg><div class="tip"></div></div>'
    )


def _rs_axis(v: float) -> str:
    return f"₹{v / 1e5:.1f}L" if abs(v) >= 1e5 else f"₹{v:,.0f}"


def render_html(r: BacktestResult) -> str:
    eq = _downsample(r.equity)
    bench = r.benchmark.reindex(eq.index)
    x = list(eq.index)
    equity_chart = _chart(
        "Equity vs Nifty 50 (same starting capital)",
        x,
        [
            {
                "name": r.variant,
                "values": eq.round(0).tolist(),
                "color": "var(--series-1)",
                "fmt": [f"₹{v:,.0f}" for v in eq],
            },
            {
                "name": "Nifty 50",
                "values": bench.round(0).tolist(),
                "color": "var(--ref)",
                "dash": True,
                "fmt": [f"₹{v:,.0f}" if v == v else "–" for v in bench],
            },
        ],
        _rs_axis,
    )
    dd = _downsample(r.equity / r.equity.cummax() - 1)
    dd_chart = _chart(
        "Drawdown",
        list(dd.index),
        [
            {
                "name": "Drawdown",
                "values": dd.tolist(),
                "color": "var(--series-1)",
                "fmt": [f"{v:.1%}" for v in dd],
            }
        ],
        lambda v: f"{v:.1%}",
        zero_floor=True,
    )
    m = r.metrics
    tiles = "".join(
        f'<div class="tile"><span class="muted">{LABELS[k][0]}</span><b>{fmt(k, m[k])}</b></div>'
        for k in LABELS
        if k in m
    )
    trades = r.trades_frame()
    reasons = ""
    trade_rows = ""
    if not trades.empty:
        g = trades.groupby("exit_reason").agg(
            n=("net_pnl", "size"), avg_r=("r_multiple", "mean"), pnl=("net_pnl", "sum")
        )
        reasons = "".join(
            f"<tr><td>{html.escape(reason_label(k))}</td><td>{int(v.n)}</td>"
            f"<td>{v.avg_r:+.2f}R</td><td>{rs(v.pnl)}</td></tr>"
            for k, v in g.sort_values("n", ascending=False).iterrows()
        )
        for t in trades.tail(100).iloc[::-1].itertuples():
            trade_rows += (
                f"<tr><td>{t.symbol}</td><td>{t.entry_date}</td><td>{t.entry_price:,.2f}</td>"
                f"<td>{t.exit_date}</td><td>{t.avg_exit_price:,.2f}</td><td>{t.qty}</td>"
                f"<td>{rs(t.net_pnl)}</td><td>{t.r_multiple:+.2f}</td>"
                f"<td>{html.escape(reason_label(t.exit_reason))}</td></tr>"
            )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest {html.escape(r.variant)}</title><style>{CSS}</style></head>
<body><main class="viz-root">
<h1>{html.escape(r.variant)} backtest</h1>
<p class="muted">{r.start} → {r.end} · capital ₹{r.capital:,.0f} · {r.signals} signals,
{r.skips} skipped at the open, {r.open_positions} still open at the end ·
costs and {html.escape("slippage")} included · universe is today's index list
(survivorship bias: delisted and dropped stocks are missing, so results are optimistic)</p>
<div class="tiles">{tiles}</div>
{equity_chart}{dd_chart}
<h2>Exits by reason</h2><div class="scroll"><table><tr><th>Reason</th><th>Trades</th>
<th>Avg R</th><th>Net P&amp;L</th></tr>{reasons}</table></div>
<h2>Last 100 trades</h2><div class="scroll"><table><tr><th>Symbol</th><th>Entry</th><th>Price</th>
<th>Exit</th><th>Avg exit</th><th>Qty</th><th>Net P&amp;L</th><th>R</th><th>Reason</th></tr>
{trade_rows}</table></div>
</main><script>{JS}</script></body></html>"""


def write_report(r: BacktestResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"backtest-{r.variant}-{r.start}-{r.end}"
    path = out_dir / f"{stem}.html"
    path.write_text(render_html(r), encoding="utf-8")
    r.trades_frame().to_csv(out_dir / f"{stem}-trades.csv", index=False)
    pd.DataFrame({"equity": r.equity, "invested": r.invested, "nifty": r.benchmark}).to_csv(
        out_dir / f"{stem}-equity.csv", index_label="date"
    )
    summary = {
        "book": r.book,
        "variant": r.variant,
        "start": r.start,
        "end": r.end,
        "capital": r.capital,
        "metrics": r.metrics,
        "signals": r.signals,
        "skips": r.skips,
        "open_positions": r.open_positions,
    }
    (out_dir / f"{stem}-summary.json").write_text(json.dumps(clean(summary), indent=2))
    return path
