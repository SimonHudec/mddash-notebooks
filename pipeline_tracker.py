"""
Pipeline progress tracker for MD simulation notebooks.

Provides a visual in-notebook widget (ipywidgets.HTML) and writes
a machine-readable ``pipeline-state.json`` to the working directory
so that external tools (e.g. the MDDash dashboard) can monitor
sub-step progress without any coupling to notebook internals.

Usage inside a notebook::

    from pipeline_tracker import PipelineTracker

    tracker = PipelineTracker(steps=[
        ("topology", "Topology Generation", "pdb2gmx · forcefield"),
        ("box",      "Simulation Box",      "editconf · dodecahedron"),
        ...
    ])
    tracker.show()

    with tracker.step("topology"):
        gmx.pdb2gmx(...)
"""

from __future__ import annotations

import json
import os
import time as _time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import ipywidgets as widgets
from IPython.display import display

# ────────────────────────────────────────────────────────────────────
# Default GROMACS protein MD setup steps
# ────────────────────────────────────────────────────────────────────
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

# Protocol version written into pipeline-state.json
_STATE_VERSION = 1

# Name of the state file that external readers look for
STATE_FILENAME = "pipeline-state.json"


class PipelineTracker:
    """Visual pipeline tracker with optional filesystem state output.

    Parameters
    ----------
    steps : sequence of (id, label, description) tuples, optional
        Pipeline step definitions.  Defaults to :data:`GROMACS_MD_STEPS`.
    notebook_name : str, optional
        Human-readable name written into the state file.
    state_dir : str or Path, optional
        Directory where ``pipeline-state.json`` is written.
        Defaults to the current working directory.  Set to ``None``
        to disable file output entirely.
    """

    # ── CSS (inlined inside the widget for notebook portability) ───
    _CSS = """\
<style>
.pipe{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      background:#fff;border:1px solid #e2e8f0;border-radius:12px;
      padding:20px 24px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.pipe-hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.pipe-title{font-size:14px;font-weight:600;color:#0f172a;letter-spacing:-.01em}
.pipe-cnt{font-size:12px;color:#64748b;background:#f1f5f9;border-radius:999px;
          padding:2px 10px;font-weight:500}
.pipe-bar{height:3px;background:#f1f5f9;border-radius:2px;margin-bottom:18px;overflow:hidden}
.pipe-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,#22c55e,#4ade80);
           transition:width .4s ease}
.pipe-row{display:flex;align-items:flex-start;min-height:48px}
.pipe-left{display:flex;flex-direction:column;align-items:center;margin-right:12px;flex-shrink:0}
.pipe-ind{width:26px;height:26px;border-radius:50%;display:flex;
          align-items:center;justify-content:center}
.pipe-ind-pending{border:1.5px solid #e2e8f0;background:#f8fafc}
.pipe-ind-running{border:2px solid #3b82f6;background:#eff6ff}
.pipe-ind-done{background:#22c55e}
.pipe-ind-error{background:#ef4444;color:#fff;font-size:12px;font-weight:700}
.pipe-conn{width:1.5px;height:22px;margin:3px 0}
.pipe-conn-off{background:#e2e8f0}.pipe-conn-on{background:#86efac}
.pipe-mid{flex:1;padding-top:2px}
.pipe-lbl{font-size:13px;font-weight:500;line-height:1.4}
.pipe-lbl-pending{color:#94a3b8}.pipe-lbl-running{color:#1d4ed8}
.pipe-lbl-done{color:#0f172a}.pipe-lbl-error{color:#dc2626}
.pipe-desc{font-size:11.5px;color:#94a3b8;margin-top:1px}
.pipe-right{padding-top:4px;min-width:65px;text-align:right}
.pipe-badge{font-size:11px;font-weight:500;padding:1px 8px;border-radius:999px}
.pipe-bg-green{background:#dcfce7;color:#16a34a}
.pipe-bg-blue{background:#dbeafe;color:#2563eb}
.pipe-bg-red{background:#fee2e2;color:#dc2626}
.pipe-spin{width:12px;height:12px;border:2px solid #bfdbfe;border-top:2px solid #3b82f6;
           border-radius:50%;animation:pipe-s .7s linear infinite}
@keyframes pipe-s{to{transform:rotate(360deg)}}
</style>"""

    _CHECK_SVG = (
        '<svg width="14" height="14" viewBox="0 0 16 16">'
        '<path d="M6.3 11.3 3 8l1-1 2.3 2.3L11.7 4l1 1z" fill="#fff"/>'
        "</svg>"
    )

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

        # per-step mutable state
        self._status: dict[str, str] = {s[0]: "pending" for s in self._steps}
        self._elapsed: dict[str, float] = {}
        self._errors: dict[str, str] = {}

        self._widget = widgets.HTML(
            layout=widgets.Layout(width="100%", max_width="640px")
        )
        self._render()

    # ── in-notebook HTML rendering ─────────────────────────────────

    def _render(self) -> None:
        done = sum(1 for v in self._status.values() if v == "done")
        total = len(self._steps)
        pct = round(done / total * 100)

        rows = ""
        for i, (key, label, desc) in enumerate(self._steps):
            st = self._status[key]

            if st == "done":
                icon = self._CHECK_SVG
            elif st == "running":
                icon = '<div class="pipe-spin"></div>'
            elif st == "error":
                icon = "✕"
            else:
                icon = (
                    f'<span style="font-size:11px;font-weight:600;'
                    f'color:#94a3b8">{i + 1}</span>'
                )

            if st == "done":
                t = self._elapsed.get(key)
                badge = (
                    f'<span class="pipe-badge pipe-bg-green">{t:.1f}s</span>'
                    if t else ""
                )
            elif st == "running":
                badge = '<span class="pipe-badge pipe-bg-blue">running</span>'
            elif st == "error":
                badge = '<span class="pipe-badge pipe-bg-red">error</span>'
            else:
                badge = ""

            conn = ""
            if i < total - 1:
                cc = "pipe-conn-on" if st == "done" else "pipe-conn-off"
                conn = f'<div class="pipe-conn {cc}"></div>'

            rows += (
                f'<div class="pipe-row">'
                f'<div class="pipe-left">'
                f'<div class="pipe-ind pipe-ind-{st}">{icon}</div>{conn}'
                f"</div>"
                f'<div class="pipe-mid">'
                f'<div class="pipe-lbl pipe-lbl-{st}">{label}</div>'
                f'<div class="pipe-desc">{desc}</div>'
                f"</div>"
                f'<div class="pipe-right">{badge}</div>'
                f"</div>"
            )

        header_badge = (
            '<span class="pipe-cnt">✓ Complete</span>'
            if done == total
            else f'<span class="pipe-cnt">{done} / {total}</span>'
        )

        self._widget.value = (
            f"{self._CSS}"
            f'<div class="pipe">'
            f'<div class="pipe-hd">'
            f'<span class="pipe-title">Pipeline Status</span>{header_badge}'
            f"</div>"
            f'<div class="pipe-bar">'
            f'<div class="pipe-fill" style="width:{pct}%"></div>'
            f"</div>"
            f"{rows}</div>"
        )

        # also persist to disk for external readers
        self._write_state()

    # ── filesystem state output ────────────────────────────────────

    def _write_state(self) -> None:
        """Write ``pipeline-state.json`` atomically for dashboard consumption."""
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
                        "id": key,
                        "label": label,
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
            tmp.replace(path)  # atomic on POSIX; near-atomic on Windows
        except OSError:
            pass  # non-fatal: widget still works even if disk write fails

    # ── public API ─────────────────────────────────────────────────

    def show(self) -> None:
        """Display the tracker widget in the notebook."""
        display(self._widget)

    @contextmanager
    def step(self, step_id: str):
        """Context manager that tracks a pipeline step.

        Sets the step to *running*, yields, then marks it *done*
        (or *error* if an exception propagates).  Duration is recorded.
        """
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
