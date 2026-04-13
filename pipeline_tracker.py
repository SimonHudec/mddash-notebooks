"""
Pipeline progress tracker for MD simulation notebooks.

Provides a combined control + progress widget and writes
``pipeline-state.json`` for dashboard consumption.

Usage::

    from pipeline_tracker import PipelineTracker

    tracker = PipelineTracker()
    tracker.show()          # renders Run All + step list in one card

    with tracker.step("topology"):
        gmx.pdb2gmx(...)
"""

from __future__ import annotations

import json
import time as _time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import ipywidgets as widgets
from IPython.display import display, Javascript

# ──────────────────────────────────────────────────────────────
# Default step definitions
# ──────────────────────────────────────────────────────────────
GROMACS_MD_STEPS: list[tuple[str, str, str]] = [
    ("topology",   "Topology Generation", "pdb2gmx · forcefield assignment"),
    ("box",        "Simulation Box",      "editconf · dodecahedral box"),
    ("solvate",    "Solvation",           "solvate · add water molecules"),
    ("ions",       "Counterions",         "genion · neutralize system"),
    ("minimize",   "Energy Minimization", "Steepest descent optimization"),
    ("nvt",        "NVT Equilibration",   "Constant volume · 300 K thermostat"),
    ("npt",        "NPT Equilibration",   "Constant pressure · 1 bar barostat"),
    ("production", "Production MD Setup", "Generate production run input"),
]

_STATE_VERSION = 1
STATE_FILENAME = "pipeline-state.json"

# ──────────────────────────────────────────────────────────────
# shadcn/ui-inspired stylesheet
#
# Palette based on Zinc + semantic colors from shadcn defaults.
# All classes are scoped under .pt-card to avoid notebook conflicts.
# ──────────────────────────────────────────────────────────────
_STYLESHEET = """\
<style>
.pt-card {
  --radius: 8px;
  --bg: #fff;
  --border: #e4e4e7;       /* zinc-200 */
  --fg: #09090b;           /* zinc-950 */
  --muted: #71717a;        /* zinc-500 */
  --muted-bg: #f4f4f5;     /* zinc-100 */
  --accent: #18181b;       /* zinc-900 */
  --success: #16a34a;      /* green-600 */
  --success-bg: #f0fdf4;   /* green-50 */
  --info: #2563eb;         /* blue-600 */
  --info-bg: #eff6ff;      /* blue-50 */
  --destructive: #dc2626;  /* red-600 */
  --destructive-bg: #fef2f2;

  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  max-width: 540px;
}

/* header row */
.pt-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}
.pt-title { font-size: 14px; font-weight: 600; color: var(--fg); }
.pt-counter {
  font-size: 12px; color: var(--muted); background: var(--muted-bg);
  border-radius: 999px; padding: 2px 10px; font-weight: 500;
}

/* progress bar */
.pt-progress {
  height: 2px; background: var(--muted-bg); border-radius: 1px;
  margin-bottom: 20px; overflow: hidden;
}
.pt-progress-fill {
  height: 100%; background: var(--success); border-radius: 1px;
  transition: width .35s ease;
}

/* step rows */
.pt-step { display: flex; align-items: flex-start; min-height: 40px; }
.pt-step-left {
  display: flex; flex-direction: column; align-items: center;
  margin-right: 12px; flex-shrink: 0;
}
.pt-dot {
  width: 24px; height: 24px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 600;
}
.pt-dot-pending  { border: 1.5px solid var(--border); background: var(--bg); color: var(--muted); }
.pt-dot-running  { border: 1.5px solid var(--info); background: var(--info-bg); }
.pt-dot-done     { background: var(--success); }
.pt-dot-error    { background: var(--destructive); }

.pt-line { width: 1.5px; height: 16px; margin: 2px 0; }
.pt-line-off { background: var(--border); }
.pt-line-on  { background: var(--success); opacity: .45; }

.pt-step-body { flex: 1; padding-top: 2px; }
.pt-name {
  font-size: 13px; font-weight: 500; line-height: 1.3;
}
.pt-name-pending { color: var(--muted); }
.pt-name-running { color: var(--info); }
.pt-name-done    { color: var(--fg); }
.pt-name-error   { color: var(--destructive); }
.pt-desc { font-size: 11px; color: var(--muted); margin-top: 1px; }

.pt-step-badge { padding-top: 3px; min-width: 56px; text-align: right; }

/* badges */
.pt-badge {
  display: inline-block; font-size: 11px; font-weight: 500;
  padding: 1px 8px; border-radius: var(--radius);
}
.pt-badge-done  { background: var(--success-bg); color: var(--success); }
.pt-badge-run   { background: var(--info-bg); color: var(--info); }
.pt-badge-err   { background: var(--destructive-bg); color: var(--destructive); }

/* spinner */
.pt-spin {
  width: 12px; height: 12px;
  border: 1.5px solid #bfdbfe; border-top-color: var(--info);
  border-radius: 50%; animation: pt-r .65s linear infinite;
}
@keyframes pt-r { to { transform: rotate(360deg); } }
</style>"""

_CHECK = (
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="none">'
    '<path d="M13.3 4.3 6.3 11.3 2.7 7.7" stroke="#fff" stroke-width="2"'
    ' stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


class PipelineTracker:
    """Combined Run-All button + pipeline progress card.

    Parameters
    ----------
    steps : sequence of (id, label, description) tuples
        Pipeline step definitions.  Defaults to GROMACS MD steps.
    notebook_name : str
        Label written into the JSON state file.
    state_dir : str | Path | None
        Where to write ``pipeline-state.json``.  ``None`` disables it.
    """

    def __init__(
        self,
        steps: Sequence[tuple[str, str, str]] | None = None,
        notebook_name: str = "protein-simulation-setup",
        state_dir: str | Path | None = ".",
    ):
        self._steps = list(steps or GROMACS_MD_STEPS)
        self._notebook_name = notebook_name
        self._state_dir = Path(state_dir) if state_dir is not None else None
        self._started_at = datetime.now(timezone.utc).isoformat()

        self._status: dict[str, str] = {s[0]: "pending" for s in self._steps}
        self._elapsed: dict[str, float] = {}
        self._errors: dict[str, str] = {}

        # ── Run All button (real ipywidget, not HTML) ──
        self._btn = widgets.Button(
            description="Run All Steps",
            icon="play",
            layout=widgets.Layout(width="100%", max_width="540px", height="36px"),
        )
        self._btn.add_class("pt-run-btn")
        self._btn.on_click(self._on_run_all)

        # ── Progress card (HTML) ──
        self._html = widgets.HTML(
            layout=widgets.Layout(width="100%", max_width="540px")
        )
        self._container = widgets.VBox(
            [self._btn, self._html],
            layout=widgets.Layout(width="100%", max_width="540px", gap="10px"),
        )

        self._render()

    # ── Run All handler ────────────────────────────────────────

    @staticmethod
    def _on_run_all(_btn):
        display(Javascript(
            "(function(){"
            "if(typeof Jupyter!=='undefined'&&Jupyter.notebook)"
            "{Jupyter.notebook.execute_all_cells_below();return;}"
            "try{document.querySelector("
            "'[data-command=\"notebook:run-all-below\"]').click()}"
            "catch(e){console.warn('run-all-below unavailable',e)}"
            "})()"
        ))

    # ── HTML rendering ─────────────────────────────────────────

    def _render(self) -> None:
        done = sum(1 for v in self._status.values() if v == "done")
        total = len(self._steps)
        pct = round(done / total * 100)
        running = any(v == "running" for v in self._status.values())

        # hide button once pipeline starts
        self._btn.layout.display = "none" if (done > 0 or running) else None

        rows = ""
        for i, (key, label, desc) in enumerate(self._steps):
            st = self._status[key]

            # dot content
            if st == "done":
                dot = _CHECK
            elif st == "running":
                dot = '<div class="pt-spin"></div>'
            elif st == "error":
                dot = '<span style="color:#fff;font-size:11px">✕</span>'
            else:
                dot = str(i + 1)

            # badge
            if st == "done":
                t = self._elapsed.get(key)
                badge = (
                    f'<span class="pt-badge pt-badge-done">{t:.1f}s</span>'
                    if t else ""
                )
            elif st == "running":
                badge = '<span class="pt-badge pt-badge-run">running</span>'
            elif st == "error":
                badge = '<span class="pt-badge pt-badge-err">error</span>'
            else:
                badge = ""

            # connector line
            line = ""
            if i < total - 1:
                lc = "pt-line-on" if st == "done" else "pt-line-off"
                line = f'<div class="pt-line {lc}"></div>'

            rows += (
                f'<div class="pt-step">'
                f'<div class="pt-step-left">'
                f'<div class="pt-dot pt-dot-{st}">{dot}</div>{line}</div>'
                f'<div class="pt-step-body">'
                f'<div class="pt-name pt-name-{st}">{label}</div>'
                f'<div class="pt-desc">{desc}</div></div>'
                f'<div class="pt-step-badge">{badge}</div>'
                f'</div>'
            )

        counter = (
            '<span class="pt-counter">Complete</span>'
            if done == total
            else f'<span class="pt-counter">{done}/{total}</span>'
        )

        self._html.value = (
            f'{_STYLESHEET}'
            f'<div class="pt-card">'
            f'<div class="pt-header">'
            f'<span class="pt-title">Pipeline</span>{counter}</div>'
            f'<div class="pt-progress">'
            f'<div class="pt-progress-fill" style="width:{pct}%"></div></div>'
            f'{rows}</div>'
        )
        self._write_state()

    # ── JSON state file ────────────────────────────────────────

    def _write_state(self) -> None:
        if self._state_dir is None:
            return
        try:
            state = {
                "version": _STATE_VERSION,
                "notebook": self._notebook_name,
                "started_at": self._started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "steps": [
                    {
                        "id": key, "label": label,
                        "status": self._status[key],
                        "elapsed": self._elapsed.get(key),
                        "error": self._errors.get(key),
                    }
                    for key, label, _ in self._steps
                ],
                "current_step": next(
                    (k for k in self._status if self._status[k] == "running"),
                    None,
                ),
                "error": next(iter(self._errors.values()), None),
            }
            path = self._state_dir / STATE_FILENAME
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(path)
        except OSError:
            pass

    # ── public API ─────────────────────────────────────────────

    def show(self) -> None:
        """Display the combined control + progress widget."""
        display(self._container)

    @contextmanager
    def step(self, step_id: str):
        """Context manager wrapping a pipeline step."""
        self._status[step_id] = "running"
        self._render()
        t0 = _time.time()
        try:
            yield
            self._elapsed[step_id] = _time.time() - t0
            self._status[step_id] = "done"
        except Exception as exc:
            self._elapsed[step_id] = _time.time() - t0
            self._status[step_id] = "error"
            self._errors[step_id] = str(exc)
            self._render()
            raise
        self._render()
