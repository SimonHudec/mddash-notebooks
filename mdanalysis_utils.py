"""
Pure helper functions for protein MD trajectory analysis.

No hidden global state. All functions are deterministic: same inputs -> same outputs.
Designed to be imported from a Jupyter notebook with zero side effects.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import mdtraj as md
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

# Colour-blind-safe palette for publication-ready figures.
_COLOUR_CYCLE = [
    to_rgba("#0173B2"),  # blue
    to_rgba("#DE8F05"),  # orange
    to_rgba("#029E73"),  # green
    to_rgba("#D55E00"),  # red
    to_rgba("#CC78BC"),  # purple
    to_rgba("#CA9161"),  # brown
    to_rgba("#FBAFE4"),  # pink
    to_rgba("#949494"),  # grey
]


def _publication_style() -> None:
    """Apply tight, publication-ready Matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })


def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they do not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _resolve_time_axis(traj: md.Trajectory, timestep_ps: Optional[float] = None) -> np.ndarray:
    """
    Return a time axis in nanoseconds for ``traj``.

    Args:
        traj: Input trajectory.
        timestep_ps: Timestep in ps. If not supplied, falls back to
            ``traj.timestep``.

    Returns:
        A 1-D array of time values in nanoseconds.
    """
    dt = timestep_ps if timestep_ps is not None else traj.timestep
    return np.arange(traj.n_frames) * dt / 1000.0  # ps -> ns


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #


def plot_energy(
    time_ns: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    outpath: str,
    xlabel: str = "Time (ns)",
) -> None:
    """
    Create a single-panel, publication-ready line plot and save it.

    Args:
        time_ns: 1-D array of time values in nanoseconds.
        values: 1-D array of the quantity to plot.
        title: Figure title.
        ylabel: Y-axis label.
        outpath: Where to write the figure, e.g.
            ``analysis_output/temperature.png``.
        xlabel: X-axis label. Defaults to ``"Time (ns)"``.
    """
    _publication_style()
    _ensure_dir(outpath)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(time_ns, values, color=_COLOUR_CYCLE[0], linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
    fig.savefig(outpath)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Trajectory preprocessing
# --------------------------------------------------------------------------- #


def align_trajectory(
    traj: md.Trajectory,
    reference_frame: int = 0,
    atom_selection: str = "protein",
) -> md.Trajectory:
    """
    Center and align a trajectory to a reference frame.

    The protein (or the atoms matching ``atom_selection``) is used for the
    alignment superposition. The returned trajectory is a **new** object; the
    input is never modified in place.

    Args:
        traj: Input trajectory.
        reference_frame: Frame index to align against.
        atom_selection: MDTraj DSL selection string used for the superposition.
            Defaults to ``"protein"``.

    Returns:
        Aligned trajectory.

    Raises:
        ValueError: If ``atom_selection`` returns no atoms.
    """
    # mdtraj.superpose modifies in place, so work on a copy.
    aligned = traj.slice(range(traj.n_frames))
    aligned.superpose(
        traj,
        frame=reference_frame,
        atom_indices=traj.topology.select(atom_selection),
    )
    # Center the aligned selection at the origin for nicer visualisation.
    atom_indices = aligned.topology.select(atom_selection)
    if len(atom_indices) == 0:
        available = sorted({r.name for r in aligned.topology.residues})
        raise ValueError(f"Selection '{atom_selection}' returned no atoms. Available residues: {available}")
    aligned.xyz[:, atom_indices, :] -= aligned.xyz[:, atom_indices, :].mean(axis=(0, 1), keepdims=True)
    return aligned


def image_molecules(traj: md.Trajectory) -> md.Trajectory:
    """
    Attempt to image molecules so they are whole across periodic boundaries.

    Falls back to returning the original trajectory if mdtraj raises (e.g. for
    non-orthogonal boxes without an anchor molecule).

    Args:
        traj: Input trajectory.

    Returns:
        Imaged trajectory, or the original if imaging is unsupported.
    """
    try:
        return traj.image_molecules()
    except Exception:  # noqa: BLE001
        # mdtraj raises a variety of exceptions here.
        return traj


def save_aligned_files(
    traj: md.Trajectory,
    structure_path: str,
    trajectory_path: str,
) -> None:
    """
    Save aligned topology and trajectory for Molstar or any other viewer.

    Args:
        traj: Aligned trajectory.
        structure_path: Path for the topology file, e.g.
            ``analysis_output/protein-aligned.gro``.
        trajectory_path: Path for the trajectory file, e.g.
            ``analysis_output/protein-aligned.xtc``.
    """
    _ensure_dir(structure_path)
    _ensure_dir(trajectory_path)
    traj[0].save_gro(structure_path)
    traj.save_xtc(trajectory_path)


# --------------------------------------------------------------------------- #
# Geometric measurements
# --------------------------------------------------------------------------- #


def _resolve_atom_indices(
    traj: md.Trajectory,
    selections: List[str],
) -> np.ndarray:
    """
    Resolve a list of MDTraj DSL selections to atom indices.

    Args:
        traj: Input trajectory.
        selections: List of MDTraj DSL selection strings.

    Returns:
        Array of atom index arrays.

    Raises:
        ValueError: If any selection is empty.
    """
    indices = np.array([traj.topology.select(sel) for sel in selections])
    for sel, idx in zip(selections, indices):
        if len(idx) == 0:
            available = sorted({r.name for r in traj.topology.residues})
            raise ValueError(f"Selection '{sel}' returned no atoms. Available residues: {available}")
    return indices


def measure_distance(
    traj: md.Trajectory,
    selection_pairs: List[Tuple[str, str]],
    timestep_ps: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Measure centre-of-geometry distances between atom selections.

    Each selection is reduced to its geometric centre at every frame, then the
    Euclidean distance between the two centres is returned.

    Args:
        traj: Input trajectory.
        selection_pairs: List of ``(selection_a, selection_b)`` tuples. Each
            selection is an MDTraj DSL string, e.g.
            ``"residue 10 and name CA"``.
        timestep_ps: Timestep in ps. Defaults to ``traj.timestep``.

    Returns:
        A 3-tuple of ``(time_ns, distances, labels)``.

        - ``time_ns``: 1-D array of time values in nanoseconds.
        - ``distances``: 2-D array of shape ``(n_frames, n_pairs)`` in nm.
        - ``labels``: Human-readable label for each pair.

    Raises:
        ValueError: If a pair contains an empty selection, or if no pairs are
            provided.
    """
    labels: List[str] = []
    dists_list: List[np.ndarray] = []
    for sel_a, sel_b in selection_pairs:
        idx_a = traj.topology.select(sel_a)
        idx_b = traj.topology.select(sel_b)
        if len(idx_a) == 0 or len(idx_b) == 0:
            available = sorted({r.name for r in traj.topology.residues})
            raise ValueError(
                f"Distance pair ({sel_a!r}, {sel_b!r}) contains an empty selection. Available residues: {available}"
            )
        labels.append(f"{sel_a} — {sel_b}")
        # Centre-of-geometry for each group at every frame.
        pos_a = traj.xyz[:, idx_a, :].mean(axis=1)  # (n_frames, 3)
        pos_b = traj.xyz[:, idx_b, :].mean(axis=1)  # (n_frames, 3)
        dist = np.linalg.norm(pos_a - pos_b, axis=1)  # (n_frames,)
        dists_list.append(dist)

    if not dists_list:
        raise ValueError("No distance pairs provided.")

    distances = np.column_stack(dists_list)  # (n_frames, n_pairs)
    time_ns = _resolve_time_axis(traj, timestep_ps)
    return time_ns, distances, labels


def measure_angle(
    traj: md.Trajectory,
    selection_triples: List[Tuple[str, str, str]],
    timestep_ps: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Measure angles (in degrees) between three atom selections.

    Args:
        traj: Input trajectory.
        selection_triples: Each tuple defines ``(sel_a, sel_b, sel_c)`` for
            the angle ``a-b-c``.
        timestep_ps: Timestep in ps. Defaults to ``traj.timestep``.

    Returns:
        A 3-tuple of ``(time_ns, angles, labels)``.

        - ``time_ns``: 1-D array of time values in nanoseconds.
        - ``angles``: 2-D array of shape ``(n_frames, n_angles)`` in degrees.
        - ``labels``: Human-readable label for each angle.

    Raises:
        ValueError: If a triple contains an empty selection, or if no triples
            are provided.
    """
    labels: List[str] = []
    triples: List[np.ndarray] = []
    for sel_a, sel_b, sel_c in selection_triples:
        idx = _resolve_atom_indices(traj, [sel_a, sel_b, sel_c])
        labels.append(f"{sel_a} — {sel_b} — {sel_c}")
        triples.append(idx)

    if not triples:
        raise ValueError("No angle triples provided.")

    all_triples = np.vstack(triples)
    angles = md.compute_angles(traj, all_triples)  # radians
    angles = np.degrees(angles)
    time_ns = _resolve_time_axis(traj, timestep_ps)
    return time_ns, angles, labels


def measure_dihedral(
    traj: md.Trajectory,
    selection_quads: List[Tuple[str, str, str, str]],
    timestep_ps: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Measure dihedral angles (in degrees) between four atom selections.

    Args:
        traj: Input trajectory.
        selection_quads: Each tuple defines ``(sel_a, sel_b, sel_c, sel_d)``
            for the dihedral ``a-b-c-d``.
        timestep_ps: Timestep in ps. Defaults to ``traj.timestep``.

    Returns:
        A 3-tuple of ``(time_ns, dihedrals, labels)``.

        - ``time_ns``: 1-D array of time values in nanoseconds.
        - ``dihedrals``: 2-D array of shape
          ``(n_frames, n_dihedrals)`` in degrees.
        - ``labels``: Human-readable label for each dihedral.

    Raises:
        ValueError: If a quad contains an empty selection, or if no quads are
            provided.
    """
    labels: List[str] = []
    quads: List[np.ndarray] = []
    for sel_a, sel_b, sel_c, sel_d in selection_quads:
        idx = _resolve_atom_indices(traj, [sel_a, sel_b, sel_c, sel_d])
        labels.append(f"{sel_a} — {sel_b} — {sel_c} — {sel_d}")
        quads.append(idx)

    if not quads:
        raise ValueError("No dihedral quads provided.")

    all_quads = np.vstack(quads)
    dihedrals = md.compute_dihedrals(traj, all_quads)  # radians
    dihedrals = np.degrees(dihedrals)
    time_ns = _resolve_time_axis(traj, timestep_ps)
    return time_ns, dihedrals, labels


# --------------------------------------------------------------------------- #
# Structure quality
# --------------------------------------------------------------------------- #


def dssp_timeline(
    traj: md.Trajectory,
    outpath: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run DSSP secondary-structure assignment and optionally plot a heat-map.

    Args:
        traj: Input trajectory. Must contain protein residues.
        outpath: If provided, a publication-ready heat-map is saved to this
            path.

    Returns:
        A 3-tuple of ``(time_ns, residues, ss_codes)``.

        - ``time_ns``: 1-D array of time values in nanoseconds.
        - ``residues``: 1-D array of residue indices (0-based).
        - ``ss_codes``: 2-D integer array of shape
          ``(n_frames, n_residues)`` encoding secondary structure. The
          integer mapping is:

          - ``0`` — Helix (H, G, I)
          - ``1`` — Sheet (E, B)
          - ``2`` — Coil / other (C, S, T, ' ')
    """
    dssp = md.compute_dssp(traj, simplified=True)  # (n_frames, n_residues)

    # Map three-letter DSSP codes to integers for imshow.
    mapping = {"H": 0, "G": 0, "I": 0, "E": 1, "B": 1, "C": 2, "S": 2, "T": 2, " ": 2}
    ss_codes = np.vectorize(mapping.get)(dssp)

    n_residues = dssp.shape[1]
    time_ns = _resolve_time_axis(traj)
    residues = np.arange(n_residues)

    if outpath is not None:
        _publication_style()
        _ensure_dir(outpath)

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            ss_codes.T,
            aspect="auto",
            origin="lower",
            extent=[time_ns[0], time_ns[-1], 0, n_residues],
            cmap="viridis",
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Residue index")
        ax.set_title("Secondary structure (DSSP)")

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["Helix", "Sheet", "Coil/other"])

        fig.savefig(outpath)
        plt.close(fig)

    return time_ns, residues, ss_codes


def ramachandran(
    traj: md.Trajectory,
    residue_indices: Optional[List[int]] = None,
    outpath: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract phi / psi dihedrals and optionally plot a Ramachandran map.

    Args:
        traj: Input trajectory.
        residue_indices: Restrict analysis to these 0-based residue indices.
            If ``None``, all protein residues are used.
        outpath: If provided, a publication-ready 2-D histogram is saved.

    Returns:
        A 2-tuple of ``(phi, psi)``.

        - ``phi``: Flattened array of phi angles in degrees.
        - ``psi``: Flattened array of psi angles in degrees.
    """
    # mdtraj returns (n_frames, n_residues - 1) in radians.
    phi_rad, psi_rad = md.compute_phi(traj), md.compute_psi(traj)

    phi = np.degrees(phi_rad[1].flatten())  # [1] gives the array, [0] is indices
    psi = np.degrees(psi_rad[1].flatten())

    if residue_indices is not None:
        # phi/psi arrays have shape (n_frames, n_dihedrals)
        # where n_dihedrals = n_residues - 1.
        # mdtraj.compute_phi returns dihedrals for residues 1..n-1.
        # A residue index r corresponds to phi[r-1] and psi[r-1].
        valid = [r for r in residue_indices if 1 <= r < traj.n_residues]
        dih_indices = [r - 1 for r in valid]
        phi = phi_rad[1][:, dih_indices].flatten()
        psi = psi_rad[1][:, dih_indices].flatten()
        phi = np.degrees(phi)
        psi = np.degrees(psi)

    if outpath is not None:
        _publication_style()
        _ensure_dir(outpath)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        hist, xedges, yedges = np.histogram2d(phi, psi, bins=180, range=[[-180, 180], [-180, 180]])
        # Omit the extreme singleton bins that dominate the colour scale.
        vmax = np.percentile(hist, 99)
        im = ax.imshow(
            hist.T,
            origin="lower",
            extent=[-180, 180, -180, 180],
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_xlabel(r"$\phi$ (°)")
        ax.set_ylabel(r"$\psi$ (°)")
        ax.set_title("Ramachandran plot")
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-180, 181, 60))
        fig.colorbar(im, ax=ax, label="Count")
        fig.savefig(outpath)
        plt.close(fig)

    return phi, psi