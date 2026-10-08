"""
Production VAR residual generation for GBA traffic data.

Pipeline:
    GBA traffic + road-network adjacency
            ↓
    Topology-aware VAR
            ↓
    VAR coefficients
            ↓
    Residuals + valid mask
            ↓
    Saved outputs for causal discovery
"""

import json
import time
from pathlib import Path

import h5py
import numpy as np

from causal.var import TopologyAwareVAR


# ============================================================
# CONFIGURATION
# ============================================================

TRAFFIC_FILE = Path(
    r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_his_raw_2021.h5"
)

ADJACENCY_FILE = Path(
    r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_rn_adj.npy"
)

OUTPUT_DIR = Path("outputs") / "var"

# Controlled production run
NUM_TIMESTEPS = 20_000
NUM_NODES = 1_000

LAG_ORDER = 1
RIDGE_ALPHA = 1e-5
USE_GPU = True


# ============================================================
# DATA LOADING
# ============================================================

def load_traffic(
    traffic_file: Path,
    num_timesteps: int,
    num_nodes: int,
) -> np.ndarray:
    """
    Load the requested GBA traffic subset.

    No imputation or filling is performed.
    Missing values remain NaN.
    """

    with h5py.File(traffic_file, "r") as f:
        traffic = f["traffic"][
            :num_timesteps,
            :num_nodes,
        ]

    return np.asarray(traffic, dtype=np.float32)


def load_adjacency(
    adjacency_file: Path,
    num_nodes: int,
) -> np.ndarray:
    """
    Load the corresponding road-network adjacency subset.
    """

    adjacency = np.load(adjacency_file, mmap_mode="r")

    adjacency = np.asarray(
        adjacency[:num_nodes, :num_nodes],
        dtype=np.float32,
    )

    return adjacency


# ============================================================
# OUTPUT
# ============================================================

def save_results(
    output_dir: Path,
    result,
    metadata: dict,
) -> None:
    """
    Save VAR outputs required by later causal-discovery stages.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "coefficients.npy",
        result.coefficients,
    )

    np.save(
        output_dir / "residuals.npy",
        result.residuals,
    )

    np.save(
        output_dir / "valid_mask.npy",
        result.valid_mask,
    )

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PRODUCTION TOPOLOGY-AWARE VAR")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Traffic file : {TRAFFIC_FILE}")
    print(f"  Adjacency    : {ADJACENCY_FILE}")
    print(f"  Timesteps    : {NUM_TIMESTEPS}")
    print(f"  Sensors      : {NUM_NODES}")
    print(f"  Lag order    : {LAG_ORDER}")
    print(f"  Ridge alpha  : {RIDGE_ALPHA}")
    print(f"  GPU          : {USE_GPU}")

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not TRAFFIC_FILE.exists():
        raise FileNotFoundError(
            f"Traffic file not found: {TRAFFIC_FILE}"
        )

    if not ADJACENCY_FILE.exists():
        raise FileNotFoundError(
            f"Adjacency file not found: {ADJACENCY_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading traffic data...")

    load_start = time.perf_counter()

    traffic = load_traffic(
        TRAFFIC_FILE,
        NUM_TIMESTEPS,
        NUM_NODES,
    )

    adjacency = load_adjacency(
        ADJACENCY_FILE,
        NUM_NODES,
    )

    load_time = time.perf_counter() - load_start

    print(f"Traffic shape    : {traffic.shape}")
    print(f"Adjacency shape  : {adjacency.shape}")
    print(f"Loading time     : {load_time:.3f} seconds")

    # --------------------------------------------------------
    # Missing-data diagnostics
    # --------------------------------------------------------

    missing_count = np.count_nonzero(~np.isfinite(traffic))
    total_count = traffic.size

    missing_percentage = (
        100.0 * missing_count / total_count
    )

    print("\nINPUT DATA")
    print(f"  Total values       : {total_count}")
    print(f"  Missing/non-finite : {missing_count}")
    print(f"  Missing percentage : {missing_percentage:.6f}%")

    # --------------------------------------------------------
    # Adjacency diagnostics
    # --------------------------------------------------------

    nonzero_adjacency = np.count_nonzero(adjacency)
    density = nonzero_adjacency / adjacency.size
    self_loops = np.count_nonzero(np.diag(adjacency))

    print("\nADJACENCY")
    print(f"  Non-zero entries : {nonzero_adjacency}")
    print(f"  Density          : {density:.6f}")
    print(f"  Self-loops       : {self_loops}")

    # --------------------------------------------------------
    # Create VAR model
    # --------------------------------------------------------

    print("\nCreating topology-aware VAR model...")

    model = TopologyAwareVAR(
        lag_order=LAG_ORDER,
        ridge_alpha=RIDGE_ALPHA,
        use_gpu=USE_GPU,
    )

    # --------------------------------------------------------
    # Fit VAR
    # --------------------------------------------------------

    print("\nFitting VAR...")

    start_time = time.perf_counter()

    result = model.fit(
        data=traffic,
        adjacency=adjacency,
    )

    runtime = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VAR RESULT")
    print("=" * 70)

    print(f"Runtime              : {runtime:.3f} seconds")
    print(f"Runtime              : {runtime / 60:.3f} minutes")
    print(f"Coefficient shape    : {result.coefficients.shape}")
    print(f"Residual shape       : {result.residuals.shape}")
    print(f"Valid-mask shape     : {result.valid_mask.shape}")
    print(f"Lag order            : {result.lag_order}")

    # --------------------------------------------------------
    # Residual coverage
    # --------------------------------------------------------

    valid_counts = np.sum(
        result.valid_mask,
        axis=0,
    )

    coverage = (
        valid_counts / NUM_TIMESTEPS
    ) * 100.0

    print("\nRESIDUAL COVERAGE")

    print(
        f"  Minimum valid residuals : "
        f"{valid_counts.min()}"
    )

    print(
        f"  Maximum valid residuals : "
        f"{valid_counts.max()}"
    )

    print(
        f"  Mean valid residuals    : "
        f"{valid_counts.mean():.3f}"
    )

    print(
        f"  Minimum coverage (%)    : "
        f"{coverage.min():.3f}"
    )

    print(
        f"  Maximum coverage (%)    : "
        f"{coverage.max():.3f}"
    )

    print(
        f"  Mean coverage (%)       : "
        f"{coverage.mean():.6f}"
    )

    # --------------------------------------------------------
    # Residual statistics
    # --------------------------------------------------------

    finite_residuals = result.residuals[
        np.isfinite(result.residuals)
    ]

    print("\nRESIDUAL VALUES")

    print(
        f"  Finite residual values : "
        f"{finite_residuals.size}"
    )

    print(
        f"  NaN residual values    : "
        f"{np.count_nonzero(~np.isfinite(result.residuals))}"
    )

    if finite_residuals.size > 0:

        print(
            f"  Residual mean         : "
            f"{finite_residuals.mean()}"
        )

        print(
            f"  Residual std          : "
            f"{finite_residuals.std()}"
        )

        print(
            f"  Residual minimum      : "
            f"{finite_residuals.min()}"
        )

        print(
            f"  Residual maximum      : "
            f"{finite_residuals.max()}"
        )

    # --------------------------------------------------------
    # Coefficient statistics
    # --------------------------------------------------------

    coefficients = result.coefficients

    nonzero_coefficients = np.count_nonzero(
        coefficients
    )

    print("\nCOEFFICIENTS")

    print(
        f"  Non-zero coefficients : "
        f"{nonzero_coefficients}"
    )

    print(
        f"  Minimum               : "
        f"{coefficients.min()}"
    )

    print(
        f"  Maximum               : "
        f"{coefficients.max()}"
    )

    print(
        f"  Mean                  : "
        f"{coefficients.mean()}"
    )

    print(
        f"  Std                   : "
        f"{coefficients.std()}"
    )

    # --------------------------------------------------------
    # Topology consistency
    # --------------------------------------------------------

    nonzero_adjacency = np.count_nonzero(
        adjacency
    )

    print("\nTOPOLOGY CONSISTENCY")

    print(
        f"  Non-zero adjacency entries : "
        f"{nonzero_adjacency}"
    )

    print(
        f"  Non-zero VAR coefficients  : "
        f"{nonzero_coefficients}"
    )

    topology_preserved = (
        np.all(
            result.coefficients[
                :, adjacency == 0
            ] == 0
        )
    )

    print(
        f"  Topology restriction       : "
        f"{'PRESERVED' if topology_preserved else 'VIOLATED'}"
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    metadata = {
        "traffic_file": str(TRAFFIC_FILE),
        "adjacency_file": str(ADJACENCY_FILE),
        "num_timesteps": NUM_TIMESTEPS,
        "num_nodes": NUM_NODES,
        "lag_order": LAG_ORDER,
        "ridge_alpha": RIDGE_ALPHA,
        "use_gpu": USE_GPU,
        "traffic_shape": list(traffic.shape),
        "adjacency_shape": list(adjacency.shape),
        "coefficient_shape": list(result.coefficients.shape),
        "residual_shape": list(result.residuals.shape),
        "valid_mask_shape": list(result.valid_mask.shape),
        "runtime_seconds": runtime,
        "runtime_minutes": runtime / 60.0,
        "input_missing_count": int(missing_count),
        "input_missing_percentage": float(missing_percentage),
        "mean_residual_coverage": float(coverage.mean()),
        "minimum_residual_coverage": float(coverage.min()),
        "maximum_residual_coverage": float(coverage.max()),
        "nonzero_adjacency": int(nonzero_adjacency),
        "nonzero_coefficients": int(nonzero_coefficients),
        "topology_restriction_preserved": bool(
            topology_preserved
        ),
    }

    save_results(
        OUTPUT_DIR,
        result,
        metadata,
    )

    print("\nOUTPUTS SAVED")

    print(f"  Directory      : {OUTPUT_DIR}")
    print("  coefficients   : coefficients.npy")
    print("  residuals      : residuals.npy")
    print("  valid mask     : valid_mask.npy")
    print("  metadata       : metadata.json")

    print("\n" + "=" * 70)
    print("PRODUCTION VAR RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()