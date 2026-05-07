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

Whole cells are always executed as-is during "Run All"; the wrappers only
control how progress is *labelled and reported*, never what runs.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import ipywidgets as widgets
from IPython.display import display

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

# A small inline shadcn-zinc inspired stylesheet. Scoped with a parent class
# so it does not bleed into the rest of the notebook.
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
.pt-card h3 {
  margin: 0 0 4px 0;
  font-size: 15px;
  font-weight: 600;
}
.pt-card .pt-sub {
  color: #71717a;
  font-size: 12px;
  margin-bottom: 12px;
}
.pt-bar {
  height: 6px;
  background: #f4f4f5;
  border-radius: 999px;
  overflow: hidden;
  margin-bottom: 12px;
}
.pt-bar > div {
  height: 100%;
  background: #18181b;
  transition: width 200ms ease;
}
.pt-step {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
  font-size: 13px;
}
.pt-dot {
  width: 18px;
  height: 18px;
  border-radius: 999px;
  border: 1px solid #d4d4d8;
  background: #fafafa;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.pt-step.running .pt-dot {
  border-color: #18181b;
  background: #18181b;
  color: #fff;
}
.pt-step.done .pt-dot {
  border-color: #16a34a;
  background: #16a34a;
  color: #fff;
}
.pt-step.error .pt-dot {
  border-color: #dc2626;
  background: #dc2626;
  color: #fff;
}
.pt-step.running .pt-label { font-weight: 600; }
.pt-step.pending .pt-label { color: #71717a; }
.pt-step.error .pt-label { color: #b91c1c; font-weight: 600; }
.pt-spinner {
  width: 10px; height: 10px; border-radius: 999px;
  border: 2px solid rgba(255,255,255,0.4);
  border-top-color: #fff;
  animation: pt-spin 0.8s linear infinite;
}
@keyframes pt-spin { to { transform: rotate(360deg); } }
.pt-run-btn button {
  background: #18181b !important;
  color: #fafafa !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 6px 14px !important;
  font-weight: 500 !important;
}
.pt-run-btn button:hover { background: #27272a !important; }
.pt-run-btn button:disabled { opacity: 0.5 !important; }
</style>
"""

_CHECK = "\u2713"   # ✓
_CROSS = "\u2715"   # ✕


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Make a filesystem/json-friendly id from a step label."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s or "step"


def _strip_magics(src: str) -> str:
    """
    Replace IPython magic / shell lines with blank lines so the source can be
    handed to ast.parse. We keep the line count so error messages stay sane.
    """
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def _extract_heading(md_src: str) -> Optional[str]:
    """
    Return the *last* heading line in a markdown cell, stripped of leading
    hashes. Returns None if the cell has no heading. We pick the last one so
    that a cell like '## Section\\n### Subsection' is reported as the
    deeper, more specific subsection.
    """
    last = None
    for line in md_src.splitlines():
        s = line.strip()
        if s.startswith("#"):
            last = re.sub(r"^#+\s*", "", s).strip()
    return last or None


def _find_step_calls(code_src: str) -> List[str]:
    """
    Return the list of literal labels passed to ``tracker.step("...")`` in
    the given source. Anything that's not a plain string literal is ignored.
    """
    try:
        tree = ast.parse(_strip_magics(code_src))
    except SyntaxError:
        return []

    labels: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # We are looking for <something>.step(...). The "something" is most
        # commonly the tracker variable, but we are not picky about its name.
        if not (isinstance(func, ast.Attribute) and func.attr == "step"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            labels.append(first.value)
    return labels


def _find_notebook() -> Optional[Path]:
    """
    Best-effort locate the .ipynb file we are running inside. We try the
    JUPYTER_SERVER_ROOT / NOTEBOOK environment hints first and then fall
    back to the only .ipynb in the current working directory.
    """
    # Some Jupyter setups expose this. It's not standardised.
    hint = os.environ.get("JPY_SESSION_NAME") or os.environ.get("NOTEBOOK_FILE")
    if hint and Path(hint).suffix == ".ipynb" and Path(hint).exists():
        return Path(hint)

    candidates = sorted(Path.cwd().glob("*.ipynb"))
    if len(candidates) == 1:
        return candidates[0]
    # If there are several, pick the most recently modified.
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


# ---------------------------------------------------------------------------
# Step model
# ---------------------------------------------------------------------------

class _Step:
    """One entry in the progress card."""

    def __init__(self, label: str, source: str, cell_indices: List[int]):
        self.label = label
        self.id = _slug(label)
        self.source = source            # "heading" or "explicit"
        self.cell_indices = cell_indices  # cells that belong to this step
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


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_steps(cells: list) -> List[_Step]:
    """
    Walk the notebook cells and produce the ordered list of steps.

    Algorithm:
      - Track the current heading from markdown cells. The "current" heading
        is the last heading we saw (cells without an active heading get
        a generic "Setup" label).
      - For each code cell:
          * If it contains tracker.step("X") calls, emit one step per label
            and attach the cell index to all of them. Heading is ignored.
          * Otherwise, attach the cell to a step derived from the current
            heading. If no step exists yet for that heading, create one.
      - The first code cell is by convention the tracker cell itself
        (it imports/instantiates PipelineTracker). We skip it during
        discovery so it never appears in the card.
    """
    steps: List[_Step] = []
    current_heading: Optional[str] = None
    heading_step: Optional[_Step] = None  # step currently bound to the heading

    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))

        if ctype == "markdown":
            h = _extract_heading(src)
            if h is not None:
                current_heading = h
                heading_step = None  # next code cell will create/find its step
            continue

        if ctype != "code":
            continue

        if _is_tracker_cell(src):
            # The tracker cell itself never appears in the progress list.
            continue

        explicit = _find_step_calls(src)
        if explicit:
            for label in explicit:
                steps.append(_Step(label, "explicit", [idx]))
            # An explicit cell does not also attach to its heading; the user
            # asked for finer granularity, so we honour it.
            heading_step = None
            continue

        # Heading-based fallback.
        label = current_heading or "Setup"
        if heading_step is None:
            heading_step = _Step(label, "heading", [idx])
            steps.append(heading_step)
        else:
            heading_step.cell_indices.append(idx)

    return steps


def _is_tracker_cell(src: str) -> bool:
    """Heuristic: cell that imports or instantiates PipelineTracker."""
    return ("PipelineTracker" in src) and ("import" in src or "(" in src)


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------

class PipelineTracker:
    """
    Drop-in progress card for notebook pipelines.

    Typical usage in the *first* code cell of a notebook::

        from pipeline_tracker import PipelineTracker
        tracker = PipelineTracker()
        tracker.show()

    Then either run cells normally (the card updates live for any
    ``with tracker.step("X"):`` blocks you have), or click the "Run All"
    button to execute every code cell below this one.
    """

    def __init__(self, state_file: str = STATE_FILENAME,
                 notebook_path: Optional[str] = None):
        self.state_path = Path(state_file)
        self.notebook_path = Path(notebook_path) if notebook_path \
            else _find_notebook()

        self._steps: List[_Step] = []
        self._active_step: Optional[_Step] = None
        self._displayed = False

        # Build widgets but do not display yet.
        self._html = widgets.HTML(value="")
        self._run_btn = widgets.Button(
            description="Run All",
            tooltip="Execute every code cell below this one",
        )
        self._run_btn.add_class("pt-run-btn")
        self._run_btn.on_click(self._on_run_all_click)
        self._container = widgets.VBox([self._run_btn, self._html])

    # -- public API --------------------------------------------------------

    def show(self) -> None:
        """Render (or re-render) the progress card."""
        self._discover()
        self._render()
        if not self._displayed:
            display(widgets.HTML(_STYLESHEET))
            display(self._container)
            self._displayed = True
        self._write_state()

    @contextmanager
    def step(self, label: str):
        """
        Context manager that marks ``label`` as running on entry and done
        on exit. If the step is not in our discovered list yet (e.g. the
        notebook was edited and not saved), we add it on the fly so manual
        cell execution still works.
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

    # -- internal ---------------------------------------------------------

    def _discover(self) -> None:
        """Re-read the notebook from disk and rebuild the step list."""
        if not self.notebook_path or not self.notebook_path.exists():
            self._steps = []
            return
        try:
            nb = json.loads(self.notebook_path.read_text(encoding="utf-8"))
        except Exception:
            self._steps = []
            return
        # Preserve status of already-running/done steps when re-discovering.
        old_status = {s.id: (s.status, s.error) for s in self._steps}
        self._steps = _discover_steps(nb.get("cells", []))
        for s in self._steps:
            if s.id in old_status:
                s.status, s.error = old_status[s.id]

    def _find_or_add_step(self, label: str) -> _Step:
        for s in self._steps:
            if s.label == label:
                return s
        s = _Step(label, "explicit", [])
        self._steps.append(s)
        return s

    def _mark_running(self, s: _Step) -> None:
        s.status = RUNNING
        s.started_at = time.time()
        s.error = None
        self._active_step = s
        self._render()
        self._write_state()

    def _mark_done(self, s: _Step) -> None:
        s.status = DONE
        s.finished_at = time.time()
        if self._active_step is s:
            self._active_step = None
        self._render()
        self._write_state()

    def _mark_error(self, s: _Step, exc: BaseException) -> None:
        s.status = ERROR
        s.finished_at = time.time()
        s.error = f"{type(exc).__name__}: {exc}"
        if self._active_step is s:
            self._active_step = None
        self._render()
        self._write_state()

    # -- rendering --------------------------------------------------------

    def _render(self) -> None:
        total = len(self._steps)
        done = sum(1 for s in self._steps if s.status == DONE)
        pct = int(100 * done / total) if total else 0

        rows = []
        for s in self._steps:
            rows.append(self._render_row(s))

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

    # -- Run All ----------------------------------------------------------

    def _on_run_all_click(self, _btn) -> None:
        self._run_btn.disabled = True
        try:
            self._run_all()
        finally:
            self._run_btn.disabled = False

    def _run_all(self) -> None:
        """
        Execute every code cell that comes after the tracker cell, in order.

        Each cell is dispatched to the running IPython kernel via
        ``run_cell``. This keeps imports, variables and side effects in the
        same namespace as if the user clicked Run on each cell manually.

        Cell-level steps (heading-based) are marked running just before
        their first cell and done after their last cell. Explicit
        ``tracker.step(...)`` blocks update themselves through the context
        manager, so we don't need to do anything special for them.
        """
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            print("Run All requires an IPython kernel.")
            return

        # Refresh discovery in case the notebook was edited.
        self._discover()
        # Reset all step statuses.
        for s in self._steps:
            s.status = PENDING
            s.started_at = None
            s.finished_at = None
            s.error = None
        self._render()
        self._write_state()

        if not self.notebook_path or not self.notebook_path.exists():
            print("Could not find notebook on disk; "
                  "open and save it once, then try again.")
            return

        nb = json.loads(self.notebook_path.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])

        # Map cell index -> heading-based step (for marking around the cell).
        cell_to_heading_step = {}
        for s in self._steps:
            if s.source == "heading":
                for ci in s.cell_indices:
                    cell_to_heading_step[ci] = s

        for idx, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if not src.strip():
                continue
            if _is_tracker_cell(src):
                continue

            # Heading-based step: mark running on its first cell.
            hstep = cell_to_heading_step.get(idx)
            if hstep is not None and hstep.status == PENDING:
                self._mark_running(hstep)

            try:
                result = ip.run_cell(src, store_history=False)
            except Exception as e:
                if hstep is not None:
                    self._mark_error(hstep, e)
                return

            if not result.success:
                exc = result.error_in_exec or result.error_before_exec \
                    or RuntimeError("cell failed")
                if hstep is not None:
                    self._mark_error(hstep, exc)
                # Stop the whole run on first failure.
                return

            # Heading-based step: mark done if this was its last cell.
            if hstep is not None and idx == hstep.cell_indices[-1]:
                self._mark_done(hstep)


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
            and "PipelineTracker" in "".join(cells[0].get("source", [])):
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
