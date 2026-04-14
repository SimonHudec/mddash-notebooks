"""
Pipeline progress tracker for MD simulation notebooks.

Provides a single widget card with an integrated *Run All* button
and a live step-progress display.  Writes ``pipeline-state.json``
for external dashboard consumption.

Usage inside a notebook::

    from pipeline_tracker import PipelineTracker

    tracker = PipelineTracker()
    tracker.show()          # one card: button + progress

    with tracker.step("topology"):
        gmx.pdb2gmx(...)

Inject the tracker cell into any notebook::

    python pipeline_tracker.py inject  notebook.ipynb
    python pipeline_tracker.py inject  notebook.ipynb  output.ipynb
"""

from __future__ import annotations

import json
import time as _time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import ipywidgets as widgets
from IPython.display import display

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
# Stylesheet – shadcn / zinc palette
# ──────────────────────────────────────────────────────────────
_STYLESHEET = """\
<style>
/* Button widget override – applied via .pt-run-btn class */
.pt-run-btn button,
.pt-run-btn .jupyter-button,
.pt-run-btn .widget-button {
  background: #18181b !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
               system-ui, sans-serif !important;
  cursor: pointer !important;
  transition: opacity .15s !important;
}
.pt-run-btn button:hover,
.pt-run-btn .jupyter-button:hover,
.pt-run-btn .widget-button:hover { opacity: .85 !important; }

/* Progress card */
.pt-card {
  --radius: 8px;
  --bg: #fff;
  --border: #e4e4e7;
  --fg: #09090b;
  --muted: #71717a;
  --muted-bg: #f4f4f5;
  --success: #16a34a;
  --success-bg: #f0fdf4;
  --info: #2563eb;
  --info-bg: #eff6ff;
  --destructive: #dc2626;
  --destructive-bg: #fef2f2;

  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
               system-ui, sans-serif;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  max-width: 540px;
  overflow: hidden;
  box-sizing: border-box;
}
.pt-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}
.pt-title { font-size: 14px; font-weight: 600; color: var(--fg); }
.pt-counter {
  font-size: 12px; color: var(--muted); background: var(--muted-bg);
  border-radius: 999px; padding: 2px 10px; font-weight: 500;
}
.pt-progress {
  height: 2px; background: var(--muted-bg); border-radius: 1px;
  margin-bottom: 20px; overflow: hidden;
}
.pt-progress-fill {
  height: 100%; background: var(--success); border-radius: 1px;
  transition: width .35s ease;
}
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
.pt-dot-pending { border: 1.5px solid var(--border); background: var(--bg); color: var(--muted); }
.pt-dot-running { border: 1.5px solid var(--info); background: var(--info-bg); }
.pt-dot-done    { background: var(--success); }
.pt-dot-error   { background: var(--destructive); }
.pt-line { width: 1.5px; height: 16px; margin: 2px 0; }
.pt-line-off { background: var(--border); }
.pt-line-on  { background: var(--success); opacity: .45; }
.pt-step-body { flex: 1; padding-top: 2px; min-width: 0; }
.pt-name { font-size: 13px; font-weight: 500; line-height: 1.3; }
.pt-name-pending { color: var(--muted); }
.pt-name-running { color: var(--info); }
.pt-name-done    { color: var(--fg); }
.pt-name-error   { color: var(--destructive); }
.pt-desc { font-size: 11px; color: var(--muted); margin-top: 1px; }
.pt-step-badge { padding-top: 3px; flex-shrink: 0; text-align: right; }
.pt-badge {
  display: inline-block; font-size: 11px; font-weight: 500;
  padding: 1px 8px; border-radius: var(--radius); white-space: nowrap;
}
.pt-badge-done { background: var(--success-bg); color: var(--success); }
.pt-badge-run  { background: var(--info-bg); color: var(--info); }
.pt-badge-err  { background: var(--destructive-bg); color: var(--destructive); }
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
    """Single-widget pipeline card with integrated Run-All.

    The *Run All* button executes every code cell below the tracker
    cell by reading the ``.ipynb`` file from disk and calling
    ``get_ipython().run_cell()`` for each.  No browser-side JavaScript
    is involved, so it works identically in JupyterLab, Classic
    Notebook, and JupyterHub.

    Parameters
    ----------
    steps : sequence of (id, label, description) tuples
        Pipeline step definitions.  Defaults to GROMACS MD steps.
    notebook_name : str
        Base name (without ``.ipynb``) of the notebook file.
        Used to locate the file on disk for *Run All*.
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
        self._shown = False

        # ── Run All button (ipywidgets – Python callback, no JS) ──
        self._btn = widgets.Button(
            description="Run All Steps",
            icon="play",
            layout=widgets.Layout(width="100%", height="36px"),
        )
        self._btn.style.button_color = "#18181b"
        self._btn.add_class("pt-run-btn")
        self._btn.on_click(self._on_run_all)

        # ── Progress card ──
        self._html = widgets.HTML()

        self._container = widgets.VBox(
            [self._btn, self._html],
            layout=widgets.Layout(max_width="540px"),
        )
        self._render()

    # ── Run All – kernel-side execution ────────────────────────

    def _on_run_all(self, _btn) -> None:
        """Read the notebook from disk and execute every code cell
        after the tracker cell via ``get_ipython().run_cell()``."""
        self._btn.disabled = True
        self._btn.description = "Running\u2026"

        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return

        nb_path = self._find_notebook()
        if nb_path is None:
            return

        nb = json.loads(nb_path.read_text(encoding="utf-8"))

        self._btn.layout.display = "none"

        found_tracker = False
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"]).strip()
            if not src:
                continue
            if not found_tracker:
                if "pipeline_tracker" in src or "PipelineTracker" in src:
                    found_tracker = True
                continue
            ip.run_cell(src, store_history=False)

    def _find_notebook(self) -> Path | None:
        for name in (self._notebook_name,
                     self._notebook_name.replace("-", "_")):
            p = Path(name + ".ipynb")
            if p.exists():
                return p
        for p in sorted(Path(".").glob("*.ipynb")):
            return p
        return None

    # ── HTML rendering ─────────────────────────────────────────

    def _render(self) -> None:
        done = sum(1 for v in self._status.values() if v == "done")
        total = len(self._steps)
        pct = round(done / total * 100)
        running = any(v == "running" for v in self._status.values())

        if done > 0 or running:
            self._btn.layout.display = "none"

        rows = ""
        for i, (key, label, desc) in enumerate(self._steps):
            st = self._status[key]

            if st == "done":
                dot = _CHECK
            elif st == "running":
                dot = '<div class="pt-spin"></div>'
            elif st == "error":
                dot = '<span style="color:#fff;font-size:11px">\u2715</span>'
            else:
                dot = str(i + 1)

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
            f"{_STYLESHEET}"
            f'<div class="pt-card">'
            f'<div class="pt-header">'
            f'<span class="pt-title">Pipeline</span>{counter}</div>'
            f'<div class="pt-progress">'
            f'<div class="pt-progress-fill" style="width:{pct}%"></div></div>'
            f"{rows}</div>"
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
        """Display the tracker card (idempotent)."""
        if not self._shown:
            display(self._container)
            self._shown = True

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


# ══════════════════════════════════════════════════════════════
# Notebook injection
# ══════════════════════════════════════════════════════════════
#
# The code below is for **deployment** – adding the tracker cell
# to a plain notebook that doesn't have one yet.
#
# Who runs this?
#   - During development:  you, once, from a terminal.
#   - In MDDash production: the pod startup script (init.sh)
#     that copies template notebooks onto the shared PVC.
#
# What does it do?
#   1. Reads the .ipynb JSON file.
#   2. Inserts ONE new code cell right after the first markdown
#      cell (usually the title).  That cell contains all the
#      imports, the PipelineTracker setup, and the monkey-patching
#      of gmx.* functions.
#   3. Writes the file back (or to a new path if you give one).
#
# The notebook is NOT modified in any other way.
# ══════════════════════════════════════════════════════════════

_INJECT_SOURCE = """\
import gromacs as gmx
import nglview as nv
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import wraps

from pipeline_tracker import PipelineTracker

tracker = PipelineTracker()
tracker.show()


def _wrap_gmx_step(func_name, step_id, condition=None):
    original = getattr(gmx, func_name)
    @wraps(original)
    def wrapped(*args, **kwargs):
        if condition is not None and not condition(*args, **kwargs):
            return original(*args, **kwargs)
        with tracker.step(step_id):
            return original(*args, **kwargs)
    setattr(gmx, func_name, wrapped)


if not getattr(gmx, '_pipeline_tracker_patched', False):
    _wrap_gmx_step('pdb2gmx', 'topology')
    _wrap_gmx_step('editconf', 'box')
    _wrap_gmx_step('solvate', 'solvate')
    _wrap_gmx_step('genion', 'ions')

    _wrap_gmx_step('mdrun', 'minimize',
                    condition=lambda *a, **k: k.get('deffnm') == 'em')
    _wrap_gmx_step('mdrun', 'nvt',
                    condition=lambda *a, **k: k.get('deffnm') == 'nvt')
    _wrap_gmx_step('mdrun', 'npt',
                    condition=lambda *a, **k: k.get('deffnm') == 'npt')
    _wrap_gmx_step(
        'grompp', 'production',
        condition=lambda *a, **k: os.path.basename(
            str(k.get('o', ''))) not in {
                'ions.tpr', 'em.tpr', 'nvt.tpr', 'npt.tpr'})

    gmx._pipeline_tracker_patched = True
"""


def inject_tracker(
    notebook_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Insert a tracker cell into a notebook file.

    Parameters
    ----------
    notebook_path
        Path to an existing ``.ipynb`` file.
    output_path
        Where to write.  Defaults to overwriting *notebook_path*.

    Returns
    -------
    Path
        The path that was written.
    """
    nb_path = Path(notebook_path)
    out = Path(output_path) if output_path else nb_path

    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Don't inject twice
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if "PipelineTracker" in "".join(cell["source"]):
                print(f"Tracker already present in {nb_path}")
                return out

    tracker_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["autorun"]},
        "outputs": [],
        "source": _INJECT_SOURCE.splitlines(keepends=True),
    }

    # Insert after the first markdown cell (title).
    insert = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            insert = i + 1
            break

    nb["cells"].insert(insert, tracker_cell)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "inject":
        target = sys.argv[2]
        dest = sys.argv[3] if len(sys.argv) > 3 else None
        result = inject_tracker(target, dest)
        print(f"Tracker cell injected -> {result}")
    else:
        print("Usage: python pipeline_tracker.py inject <notebook.ipynb> [output.ipynb]")
        print()
        print("Inserts a tracker + monkey-patching cell into the notebook.")
        print("If output.ipynb is omitted, the file is modified in-place.")
"""
Pipeline progress tracker for MD simulation notebooks.

Provides a single-widget card with an integrated *Run All* button
and a live step-progress display.  Writes ``pipeline-state.json``
for external dashboard consumption.

Usage inside a notebook::

    from pipeline_tracker import PipelineTracker

    tracker = PipelineTracker()
    tracker.show()          # one card: button + progress

    with tracker.step("topology"):
        gmx.pdb2gmx(...)

Inject into an arbitrary notebook from the command line::

    python pipeline_tracker.py inject notebook.ipynb
"""

from __future__ import annotations

import json
import time as _time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import ipywidgets as widgets
from IPython.display import display

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
# JS snippet executed by the Run-All HTML button's onclick.
#
# ipywidgets.HTML sets innerHTML directly (no sanitiser), so
# inline event-handler attributes work in both JupyterLab and
# Classic Notebook.  The snippet:
#   1. Selects the parent cell so "run-all-below" starts here
#   2. Tries JupyterLab 4 → JupyterLab 3 → Classic Notebook
#   3. Hides the button immediately
# ──────────────────────────────────────────────────────────────
_RUN_ALL_JS = (
    "var c=this.closest('.jp-Cell,.cell');if(c)c.click();"
    "var me=this;setTimeout(function(){"
    "var a=window.jupyterapp||window.jupyterlab;"
    "if(a&&a.commands){a.commands.execute('notebook:run-all-below');}"
    "else if(typeof Jupyter!=='undefined'&&Jupyter.notebook)"
    "{Jupyter.notebook.execute_all_cells_below();}"
    "me.style.display='none';"
    "},60)"
)

# ──────────────────────────────────────────────────────────────
# Stylesheet – shadcn / zinc palette, scoped via .pt-card
# ──────────────────────────────────────────────────────────────
_STYLESHEET = """\
<style>
.pt-card {
  --radius: 8px;
  --bg: #fff;
  --border: #e4e4e7;
  --fg: #09090b;
  --muted: #71717a;
  --muted-bg: #f4f4f5;
  --accent: #18181b;
  --success: #16a34a;
  --success-bg: #f0fdf4;
  --info: #2563eb;
  --info-bg: #eff6ff;
  --destructive: #dc2626;
  --destructive-bg: #fef2f2;

  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  max-width: 540px;
  overflow: hidden;
  box-sizing: border-box;
}

/* Run All button */
.pt-btn {
  width: 100%; padding: 9px 16px;
  border: none; border-radius: var(--radius);
  background: var(--accent); color: #fff;
  font-size: 13px; font-weight: 500;
  cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  margin-bottom: 16px;
  font-family: inherit;
  transition: opacity .15s;
}
.pt-btn:hover { opacity: .85; }

.pt-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}
.pt-title { font-size: 14px; font-weight: 600; color: var(--fg); }
.pt-counter {
  font-size: 12px; color: var(--muted); background: var(--muted-bg);
  border-radius: 999px; padding: 2px 10px; font-weight: 500;
}

.pt-progress {
  height: 2px; background: var(--muted-bg); border-radius: 1px;
  margin-bottom: 20px; overflow: hidden;
}
.pt-progress-fill {
  height: 100%; background: var(--success); border-radius: 1px;
  transition: width .35s ease;
}

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

.pt-step-body { flex: 1; padding-top: 2px; min-width: 0; }
.pt-name { font-size: 13px; font-weight: 500; line-height: 1.3; }
.pt-name-pending { color: var(--muted); }
.pt-name-running { color: var(--info); }
.pt-name-done    { color: var(--fg); }
.pt-name-error   { color: var(--destructive); }
.pt-desc { font-size: 11px; color: var(--muted); margin-top: 1px; }

.pt-step-badge { padding-top: 3px; flex-shrink: 0; text-align: right; }

.pt-badge {
  display: inline-block; font-size: 11px; font-weight: 500;
  padding: 1px 8px; border-radius: var(--radius);
}
.pt-badge-done { background: var(--success-bg); color: var(--success); }
.pt-badge-run  { background: var(--info-bg); color: var(--info); }
.pt-badge-err  { background: var(--destructive-bg); color: var(--destructive); }

.pt-spin {
  width: 12px; height: 12px;
  border: 1.5px solid #bfdbfe; border-top-color: var(--info);
  border-radius: 50%; animation: pt-r .65s linear infinite;
}
@keyframes pt-r { to { transform: rotate(360deg); } }
</style>"""

_PLAY_ICON = (
    '<svg width="14" height="14" viewBox="0 0 16 16" fill="none">'
    '<path d="M4 3l9 5-9 5V3z" fill="currentColor"/></svg>'
)

_CHECK = (
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="none">'
    '<path d="M13.3 4.3 6.3 11.3 2.7 7.7" stroke="#fff" stroke-width="2"'
    ' stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


class PipelineTracker:
    """Single-widget pipeline card with integrated Run-All button.

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

        self._shown = False
        self._widget = widgets.HTML(
            layout=widgets.Layout(max_width="540px"),
        )
        self._render()

    # ── HTML rendering ─────────────────────────────────────────

    def _render(self) -> None:
        done = sum(1 for v in self._status.values() if v == "done")
        total = len(self._steps)
        pct = round(done / total * 100)
        running = any(v == "running" for v in self._status.values())

        # Button visible only before pipeline starts
        if done == 0 and not running:
            btn = (
                f'<button class="pt-btn" onclick="{_RUN_ALL_JS}">'
                f'{_PLAY_ICON} Run All Steps</button>'
            )
        else:
            btn = ""

        rows = ""
        for i, (key, label, desc) in enumerate(self._steps):
            st = self._status[key]

            if st == "done":
                dot = _CHECK
            elif st == "running":
                dot = '<div class="pt-spin"></div>'
            elif st == "error":
                dot = '<span style="color:#fff;font-size:11px">\u2715</span>'
            else:
                dot = str(i + 1)

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

        self._widget.value = (
            f'{_STYLESHEET}'
            f'<div class="pt-card">'
            f'{btn}'
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
        """Display the tracker widget (idempotent)."""
        if not self._shown:
            display(self._widget)
            self._shown = True

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


# ──────────────────────────────────────────────────────────────
# Notebook injection utility
# ──────────────────────────────────────────────────────────────

#: Source code injected as the first code cell.
_INJECT_SOURCE = """\
import gromacs as gmx
import nglview as nv
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import wraps

from pipeline_tracker import PipelineTracker

tracker = PipelineTracker()
tracker.show()


def _wrap_gmx_step(func_name, step_id, condition=None):
    original = getattr(gmx, func_name)
    @wraps(original)
    def wrapped(*args, **kwargs):
        if condition is not None and not condition(*args, **kwargs):
            return original(*args, **kwargs)
        with tracker.step(step_id):
            return original(*args, **kwargs)
    setattr(gmx, func_name, wrapped)


if not getattr(gmx, '_pipeline_tracker_patched', False):
    _wrap_gmx_step('pdb2gmx', 'topology')
    _wrap_gmx_step('editconf', 'box')
    _wrap_gmx_step('solvate', 'solvate')
    _wrap_gmx_step('genion', 'ions')

    _wrap_gmx_step('mdrun', 'minimize',
                    condition=lambda *a, **k: k.get('deffnm') == 'em')
    _wrap_gmx_step('mdrun', 'nvt',
                    condition=lambda *a, **k: k.get('deffnm') == 'nvt')
    _wrap_gmx_step('mdrun', 'npt',
                    condition=lambda *a, **k: k.get('deffnm') == 'npt')
    _wrap_gmx_step(
        'grompp', 'production',
        condition=lambda *a, **k: os.path.basename(
            str(k.get('o', ''))) not in {
                'ions.tpr', 'em.tpr', 'nvt.tpr', 'npt.tpr'})

    gmx._pipeline_tracker_patched = True
"""


def inject_tracker(notebook_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Inject a pipeline-tracker cell into a notebook.

    Inserts the tracker + monkey-patching code as the **first code cell**
    (right after the initial title / markdown cell).  Existing cells are
    left untouched.

    Parameters
    ----------
    notebook_path : path
        ``.ipynb`` file to modify.
    output_path : path, optional
        Where to write.  Defaults to overwriting *notebook_path*.

    Returns
    -------
    Path
        The path that was written.
    """
    nb_path = Path(notebook_path)
    out = Path(output_path) if output_path else nb_path

    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    tracker_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["autorun"]},
        "outputs": [],
        "source": _INJECT_SOURCE.splitlines(keepends=True),
    }

    # Insert after the first markdown cell (usually the title).
    insert = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            insert = i + 1
            break

    nb["cells"].insert(insert, tracker_cell)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "inject":
        target = sys.argv[2]
        dest = sys.argv[3] if len(sys.argv) > 3 else None
        result = inject_tracker(target, dest)
        print(f"Injected tracker cell into {result}")
    else:
        print("Usage: python pipeline_tracker.py inject <notebook.ipynb> [output.ipynb]")
