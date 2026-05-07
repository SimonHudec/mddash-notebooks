"""
pipeline_tracker.py

A small progress-tracking widget for Jupyter notebooks that run molecular
dynamics (or any sequential) pipelines. It shows a card with a "Run All"
button and a list of pipeline steps that update as the notebook executes,
and writes a JSON state file that can be picked up by an external dashboard
(MD Dashboard).

The tracker discovers the steps of the pipeline directly from the notebook
file, with two complementary mechanisms:

1. Markdown headings define the default step structure. Every code cell is
   attached to the most recent heading above it, and that heading becomes
   the step label.

2. Authors can override or refine this by wrapping code in a
   ``with tracker.step("name"):`` block. Wrapped blocks become explicit
   steps and take priority over the heading for the cells they appear in.

Cells are executed by the Jupyter frontend itself: the "Run All" button
dispatches a JavaScript command that triggers the standard
``notebook:run-all-cells`` action. Progress is tracked via IPython kernel
events (``pre_run_cell`` / ``post_run_cell``), so cell outputs, widgets
and errors land in their own cells, exactly as if the user pressed Run
on each one manually.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import Javascript, display

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_FILENAME = "pipeline-state.json"
_STATE_VERSION = 1

# Statuses used both internally and in the JSON state file.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
ERROR = "error"

# A small shadcn-zinc inspired stylesheet, scoped under .pt-card.
_STYLESHEET = """
<style>
.pt-card {
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  background: #ffffff;
  color: #18181b;
  border: 1px solid #e4e4e7;
  border-radius: 12px;
  padding: 16px 18px;
  margin: 6px 0;
  max-width: 720px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.pt-card h3 { margin: 0 0 4px 0; font-size: 15px; font-weight: 600; }
.pt-card .pt-sub { color: #71717a; font-size: 12px; margin-bottom: 12px; }
.pt-bar {
  height: 6px; background: #f4f4f5; border-radius: 999px;
  overflow: hidden; margin-bottom: 12px;
}
.pt-bar > div {
  height: 100%; background: #18181b; transition: width 200ms ease;
}
.pt-step {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 0; font-size: 13px;
}
.pt-dot {
  width: 18px; height: 18px; border-radius: 999px;
  border: 1px solid #d4d4d8; background: #fafafa;
  display: inline-flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.pt-step.running .pt-dot { border-color: #18181b; background: #18181b; color: #fff; }
.pt-step.done    .pt-dot { border-color: #16a34a; background: #16a34a; color: #fff; }
.pt-step.error   .pt-dot { border-color: #dc2626; background: #dc2626; color: #fff; }
.pt-step.running .pt-label { font-weight: 600; }
.pt-step.pending .pt-label { color: #71717a; }
.pt-step.error   .pt-label { color: #b91c1c; font-weight: 600; }
.pt-spinner {
  width: 10px; height: 10px; border-radius: 999px;
  border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff;
  animation: pt-spin 0.8s linear infinite;
}
@keyframes pt-spin { to { transform: rotate(360deg); } }
.pt-run-btn button {
  background: #18181b !important; color: #fafafa !important;
  border: none !important; border-radius: 8px !important;
  padding: 6px 14px !important; font-weight: 500 !important;
}
.pt-run-btn button:hover { background: #27272a !important; }
.pt-run-btn button:disabled { opacity: 0.5 !important; }
.pt-run-btn-secondary button {
  background: #ffffff !important; color: #18181b !important;
  border: 1px solid #d4d4d8 !important; border-radius: 8px !important;
  padding: 6px 14px !important; font-weight: 500 !important;
}
.pt-run-btn-secondary button:hover { background: #f4f4f5 !important; }
.pt-run-btn-secondary button:disabled { opacity: 0.5 !important; }
</style>
"""

_CHECK = "\u2713"   # ✓
_CROSS = "\u2715"   # ✕


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Make a filesystem/json-friendly id from a step label."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s or "step"


def _strip_magics(src: str) -> str:
    """Replace IPython shell/magic lines with blanks so ast.parse accepts the cell."""
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def _extract_heading(md_src: str) -> Optional[str]:
    """Return the deepest (last) heading in a markdown cell, or None."""
    last = None
    for line in md_src.splitlines():
        s = line.strip()
        if s.startswith("#"):
            last = re.sub(r"^#+\s*", "", s).strip()
    return last or None


def _find_step_calls(code_src: str) -> List[str]:
    """Return literal labels passed to tracker.step(\"...\") in this cell."""
    try:
        tree = ast.parse(_strip_magics(code_src))
    except SyntaxError:
        return []
    labels: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "step"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            labels.append(first.value)
    return labels


def _is_tracker_cell(src: str) -> bool:
    """True if this code cell is the tracker bootstrap cell itself."""
    return ("from pipeline_tracker" in src) and ("PipelineTracker" in src)


def _find_notebook() -> Optional[Path]:
    """Best-effort: locate the .ipynb we are running inside."""
    hint = os.environ.get("JPY_SESSION_NAME") or os.environ.get("NOTEBOOK_FILE")
    if hint and Path(hint).suffix == ".ipynb" and Path(hint).exists():
        return Path(hint)
    candidates = sorted(Path.cwd().glob("*.ipynb"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


# ---------------------------------------------------------------------------
# Step model
# ---------------------------------------------------------------------------

class _Step:
    def __init__(self, label: str, source: str, cell_indices: List[int]):
        self.label = label
        self.id = _slug(label)
        self.source = source            # "heading" or "explicit"
        self.cell_indices = list(cell_indices)
        self.status = PENDING
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "source": self.source,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


def _discover_steps(cells: list) -> List[_Step]:
    """
    Walk the notebook cells and produce the ordered list of steps.

    Rules:
      - The tracker bootstrap cell is skipped entirely.
      - Markdown headings update the current heading context.
      - A code cell containing tracker.step("X") becomes one explicit step
        per label and overrides the heading context for that cell.
      - Otherwise the cell is attached to a step derived from the current
        heading, creating that step if it does not exist yet.
    """
    steps: List[_Step] = []
    current_heading: Optional[str] = None
    heading_step: Optional[_Step] = None

    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))

        if ctype == "markdown":
            h = _extract_heading(src)
            if h is not None:
                current_heading = h
                heading_step = None
            continue

        if ctype != "code":
            continue
        if _is_tracker_cell(src):
            continue

        explicit = _find_step_calls(src)
        if explicit:
            for label in explicit:
                steps.append(_Step(label, "explicit", [idx]))
            heading_step = None
            continue

        label = current_heading or "Setup"
        if heading_step is None:
            heading_step = _Step(label, "heading", [idx])
            steps.append(heading_step)
        else:
            heading_step.cell_indices.append(idx)

    return steps


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------

# Module-level singleton. Re-running the tracker cell returns the same
# instance, so the widget reference, the kernel hooks and the JS dispatch
# stay in sync across re-executions.
_INSTANCE: Optional["PipelineTracker"] = None


class PipelineTracker:
    """
    Drop-in progress card for notebook pipelines.

    Typical usage in the *first* code cell of a notebook::

        from pipeline_tracker import PipelineTracker
        tracker = PipelineTracker()
        tracker.show()
    """

    # Singleton: the second call returns the first instance.
    def __new__(cls, *args, **kwargs):
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = super().__new__(cls)
            _INSTANCE._initialised = False
        return _INSTANCE

    def __init__(self, state_file: str = STATE_FILENAME,
                 notebook_path: Optional[str] = None):
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        print("PipelineTracker.__init__: start", flush=True)
        self.state_path = Path(state_file)
        print("PipelineTracker.__init__: locating notebook...", flush=True)
        self.notebook_path = Path(notebook_path) if notebook_path \
            else _find_notebook()
        print(f"PipelineTracker.__init__: notebook = {self.notebook_path}",
              flush=True)

        self._steps: List[_Step] = []
        self._cell_to_step: Dict[str, _Step] = {}  # normalised src -> step
        self._displayed = False
        self._hooks_registered = False

        print("PipelineTracker.__init__: building widgets...", flush=True)
        # The Output widget is needed so that Javascript() dispatched from
        # the button callback has a display context capable of rendering
        # the application/javascript MIME type. Without it the JS is
        # rendered as the text repr of the Javascript object and never
        # executed.
        self._html = widgets.HTML(value="")
        self._js_out = widgets.Output()
        self._run_btn = widgets.Button(
            description="Run All",
            tooltip="Execute every cell of the pipeline",
        )
        self._run_btn.add_class("pt-run-btn")
        self._run_btn.on_click(self._on_run_all_click)

        self._run_btn_kernel = widgets.Button(
            description="Run All (kernel)",
            tooltip=(
                "Run every cell from inside the kernel. Use this when the "
                "frontend Run All does not start (e.g. JupyterLab without "
                "expose_app_in_browser). Cell outputs all land in this cell."
            ),
        )
        self._run_btn_kernel.add_class("pt-run-btn-secondary")
        self._run_btn_kernel.on_click(self._on_run_all_kernel_click)

        self._buttons = widgets.HBox([self._run_btn, self._run_btn_kernel])
        self._container = widgets.VBox(
            [self._buttons, self._html, self._js_out]
        )

        print("PipelineTracker.__init__: registering kernel hooks...",
              flush=True)
        self._register_hooks()
        print("PipelineTracker.__init__: done", flush=True)

    # -- public API --------------------------------------------------------

    def show(self) -> None:
        """Render (or re-render) the progress card. Idempotent."""
        print("PipelineTracker: discovering steps...", flush=True)
        self._discover()
        print(f"PipelineTracker: {len(self._steps)} steps discovered "
              f"from {self.notebook_path}", flush=True)
        self._render()
        if not self._displayed:
            display(widgets.HTML(_STYLESHEET))
            display(self._container)
            self._displayed = True
        self._write_state()
        print("PipelineTracker: ready.", flush=True)

    @contextmanager
    def step(self, label: str):
        """
        Context manager: mark ``label`` running on entry, done on exit
        (or error if the block raises).
        """
        s = self._find_or_add_step(label)
        self._mark_running(s)
        try:
            yield s
        except Exception as e:
            self._mark_error(s, e)
            raise
        else:
            self._mark_done(s)

    # -- discovery / lookup ----------------------------------------------

    def _discover(self) -> None:
        """Re-read the notebook from disk and rebuild the step list."""
        if not self.notebook_path or not self.notebook_path.exists():
            self._steps = []
            self._cell_to_step = {}
            return
        try:
            nb = json.loads(self.notebook_path.read_text(encoding="utf-8"))
        except Exception:
            self._steps = []
            self._cell_to_step = {}
            return

        # Preserve status of already-known steps when re-discovering.
        old_status = {s.id: (s.status, s.started_at, s.finished_at, s.error)
                      for s in self._steps}
        self._steps = _discover_steps(nb.get("cells", []))
        for s in self._steps:
            if s.id in old_status:
                s.status, s.started_at, s.finished_at, s.error = old_status[s.id]

        # Build a lookup from cell source -> step, used by kernel hooks to
        # find which step a running cell belongs to.
        cells = nb.get("cells", [])
        self._cell_to_step = {}
        for s in self._steps:
            for ci in s.cell_indices:
                if 0 <= ci < len(cells):
                    src = "".join(cells[ci].get("source", []))
                    key = src.strip()
                    if key:
                        self._cell_to_step[key] = s

    def _find_or_add_step(self, label: str) -> _Step:
        for s in self._steps:
            if s.label == label:
                return s
        s = _Step(label, "explicit", [])
        self._steps.append(s)
        return s

    # -- step state transitions ------------------------------------------

    def _mark_running(self, s: _Step) -> None:
        if s.status == RUNNING:
            return
        s.status = RUNNING
        s.started_at = time.time()
        s.error = None
        self._render()
        self._write_state()

    def _mark_done(self, s: _Step) -> None:
        s.status = DONE
        s.finished_at = time.time()
        self._render()
        self._write_state()

    def _mark_error(self, s: _Step, exc: BaseException) -> None:
        s.status = ERROR
        s.finished_at = time.time()
        s.error = f"{type(exc).__name__}: {exc}"
        self._render()
        self._write_state()

    # -- kernel event hooks ----------------------------------------------

    def _register_hooks(self) -> None:
        """Subscribe to IPython cell-execution events. Idempotent."""
        if self._hooks_registered:
            return
        ip = get_ipython()
        if ip is None:
            return
        ip.events.register("pre_run_cell", self._pre_run_cell)
        ip.events.register("post_run_cell", self._post_run_cell)
        self._hooks_registered = True

    def _pre_run_cell(self, info) -> None:
        """Fired by IPython just before a cell starts executing."""
        src = (getattr(info, "raw_cell", "") or "").strip()
        if not src or _is_tracker_cell(src):
            return
        s = self._cell_to_step.get(src)
        if s is None:
            return
        if s.status in (PENDING, ERROR):
            self._mark_running(s)

    def _post_run_cell(self, result) -> None:
        """Fired by IPython after a cell finishes (success or failure)."""
        info = getattr(result, "info", None)
        src = (getattr(info, "raw_cell", "") or "").strip() if info else ""
        if not src or _is_tracker_cell(src):
            return
        s = self._cell_to_step.get(src)
        if s is None:
            return

        err = getattr(result, "error_in_exec", None) \
            or getattr(result, "error_before_exec", None)
        if err is not None:
            self._mark_error(s, err)
            return

        # A heading-based step can span multiple cells. Mark it done only
        # once its last cell has finished. We approximate "last cell" by
        # taking the largest cell index recorded for the step.
        last_idx = max(s.cell_indices) if s.cell_indices else -1
        this_idx = self._index_of_source(src)
        if this_idx == last_idx:
            self._mark_done(s)

    def _index_of_source(self, src: str) -> int:
        """Return the cell index whose source matches src, or -1."""
        if not self.notebook_path or not self.notebook_path.exists():
            return -1
        try:
            nb = json.loads(self.notebook_path.read_text(encoding="utf-8"))
        except Exception:
            return -1
        for i, c in enumerate(nb.get("cells", [])):
            if c.get("cell_type") != "code":
                continue
            if "".join(c.get("source", [])).strip() == src:
                return i
        return -1

    # -- rendering --------------------------------------------------------

    def _render(self) -> None:
        total = len(self._steps)
        done = sum(1 for s in self._steps if s.status == DONE)
        pct = int(100 * done / total) if total else 0

        rows = [self._render_row(s) for s in self._steps]

        sub = f"{done} / {total} steps complete" if total else \
              "No steps discovered yet."
        if any(s.status == ERROR for s in self._steps):
            sub = "Pipeline stopped on error."

        self._html.value = (
            f'<div class="pt-card">'
            f'  <h3>Pipeline progress</h3>'
            f'  <div class="pt-sub">{sub}</div>'
            f'  <div class="pt-bar"><div style="width:{pct}%"></div></div>'
            f'  {"".join(rows)}'
            f'</div>'
        )

    def _render_row(self, s: _Step) -> str:
        if s.status == DONE:
            mark = _CHECK
        elif s.status == ERROR:
            mark = _CROSS
        elif s.status == RUNNING:
            mark = '<span class="pt-spinner"></span>'
        else:
            mark = ""

        label = s.label
        if s.status == ERROR and s.error:
            label = f"{s.label} &mdash; {s.error}"

        return (
            f'<div class="pt-step {s.status}">'
            f'  <span class="pt-dot">{mark}</span>'
            f'  <span class="pt-label">{label}</span>'
            f'</div>'
        )

    # -- state file -------------------------------------------------------

    def _write_state(self) -> None:
        data = {
            "version": _STATE_VERSION,
            "notebook": str(self.notebook_path) if self.notebook_path else None,
            "updated_at": time.time(),
            "steps": [s.to_dict() for s in self._steps],
        }
        try:
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except Exception:
            # State file is best-effort; never break the pipeline over it.
            pass

    # -- Run All (frontend-driven) ----------------------------------------

    def _on_run_all_click(self, _btn) -> None:
        """
        Reset all step statuses, then ask the Jupyter frontend to run
        every cell in the notebook. Cells execute exactly as if the user
        clicked Run on each one; our IPython hooks update progress as
        they fire.
        """
        self._reset_steps()

        # Dispatch to the frontend. Works in JupyterLab and Notebook 7.
        # We must route this through an Output widget; otherwise the
        # callback has no notebook cell to attach the JS display to.
        self._js_out.clear_output()
        with self._js_out:
            display(Javascript(_RUN_ALL_JS))

    # -- Run All (kernel-side fallback) -----------------------------------

    def _on_run_all_kernel_click(self, _btn) -> None:
        """
        Fallback: execute every code cell directly through the kernel.
        Used in environments where the frontend command API is not
        reachable (e.g. JupyterLab without ``expose_app_in_browser``).
        All cell outputs (prints, widgets, errors) appear inside this
        cell rather than in their own cells.
        """
        self._reset_steps()
        self._run_btn.disabled = True
        self._run_btn_kernel.disabled = True
        try:
            self._run_all_kernel()
        finally:
            self._run_btn.disabled = False
            self._run_btn_kernel.disabled = False

    def _reset_steps(self) -> None:
        self._discover()
        for s in self._steps:
            s.status = PENDING
            s.started_at = None
            s.finished_at = None
            s.error = None
        self._render()
        self._write_state()

    def _run_all_kernel(self) -> None:
        ip = get_ipython()
        if ip is None:
            print("Kernel-side Run All requires an IPython kernel.")
            return
        if not self.notebook_path or not self.notebook_path.exists():
            print("Could not locate the notebook on disk.")
            return
        try:
            nb = json.loads(self.notebook_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed to read notebook: {e}")
            return
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if not src.strip() or _is_tracker_cell(src):
                continue
            # The pre/post_run_cell hooks will update step status as we go.
            result = ip.run_cell(src, store_history=False)
            if not result.success:
                # The hook has already marked the failing step as error;
                # stop the rest of the run.
                return


# Triggers the standard "Run all cells" action in the active notebook.
# In modern JupyterLab the application instance is only exposed on
# ``window.jupyterapp`` when the admin sets
# ``LabApp.expose_app_in_browser = True``. On managed JupyterHub
# deployments that flag is usually off, so we also try a DOM fallback:
# clicking the toolbar's "restart kernel and run all cells" button,
# which is reachable without the app singleton.
_RUN_ALL_JS = """
(async function () {
  function log(msg) { console.log('PipelineTracker:', msg); }

  function findByDataCommand() {
    const sels = [
      '[data-command="notebook:run-all-cells"]',
      '[data-command="runmenu:run-all"]',
      '[data-jp-command-id="notebook:run-all-cells"]',
      '[data-jp-command-id="runmenu:run-all"]',
    ];
    for (const sel of sels) {
      const el = document.querySelector(sel);
      if (el) { log('matched ' + sel); return el; }
    }
    return null;
  }

  function findByLabel() {
    // Walk every clickable element and look for "run all" in its
    // aria-label, title, or text content. Prefer plain "Run All Cells"
    // and explicitly skip "Restart..." which opens a confirm dialog.
    const all = document.querySelectorAll(
      'button, jp-button, [role="menuitem"], li.lm-Menu-item'
    );
    let fallback = null;
    for (const el of all) {
      // Never match the tracker's own buttons (would self-trigger).
      if (el.closest('.pt-run-btn') || el.closest('.pt-run-btn-secondary')) {
        continue;
      }
      const txt = (
        (el.getAttribute('aria-label') || '') + ' ' +
        (el.getAttribute('title') || '') + ' ' +
        (el.textContent || '')
      ).toLowerCase();
      if (!txt.includes('run all')) continue;
      if (txt.includes('restart')) continue;          // skip "Restart & Run All"
      if (txt.includes('above') || txt.includes('below') ||
          txt.includes('selected')) continue;
      log('matched by label: ' + txt.trim().slice(0, 80));
      return el;
    }
    return fallback;
  }

  try {
    // 1. Lab app singleton (only when expose_app_in_browser=True).
    const app =
      window.jupyterapp ||
      window.jupyterlab ||
      (window._JUPYTERLAB && window._JUPYTERLAB.application);
    if (app && app.commands) {
      log('using app.commands.execute');
      await app.commands.execute('notebook:run-all-cells');
      return;
    }

    // 2. Toolbar/menu item via data-command.
    let target = findByDataCommand();

    // 3. Walk DOM by label.
    if (!target) target = findByLabel();

    if (target) {
      log('clicking target ' + target.tagName);
      target.click();
      return;
    }

    // 4. Classic Notebook fallback.
    if (window.Jupyter && Jupyter.notebook) {
      log('using classic Jupyter.notebook');
      Jupyter.notebook.execute_all_cells();
      return;
    }

    console.warn(
      'PipelineTracker: no Run-All target found. ' +
      'Use Kernel \u2192 Restart & Run All from the menu, or ' +
      'the "Run All (kernel)" button as a last resort.'
    );
  } catch (e) {
    console.error('PipelineTracker run-all failed:', e);
  }
})();
"""


# ---------------------------------------------------------------------------
# CLI helper: inject a tracker cell into a notebook from the shell.
# ---------------------------------------------------------------------------

_INJECT_SOURCE = (
    "from pipeline_tracker import PipelineTracker\n"
    "tracker = PipelineTracker()\n"
    "tracker.show()\n"
)


def inject_tracker(notebook: str) -> None:
    """Insert a tracker cell at the top of ``notebook`` if not already there."""
    p = Path(notebook)
    nb = json.loads(p.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    if cells and cells[0].get("cell_type") == "code" \
            and _is_tracker_cell("".join(cells[0].get("source", []))):
        print("Tracker cell already present.")
        return
    cells.insert(0, {
        "cell_type": "code",
        "metadata": {},
        "source": _INJECT_SOURCE.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    })
    nb["cells"] = cells
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Injected tracker cell into {p}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pipeline_tracker.py <notebook.ipynb>")
        raise SystemExit(2)
    inject_tracker(sys.argv[1])
