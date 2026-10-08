"""
E2 Experiment:
A3T-GCN + Topology-Aware VAR + PEM

Workflow:

    Traffic Data
         ↓
    Chronological Split
         ↓
    Topology-Aware VAR
         ↓
    VAR Residuals
         ↓
    Local PEM
         ↓
    Probabilistic Causal Graph
         ↓
    A3T-GCN
         ↓
    Future Traffic Prediction

Important:
- VAR and PEM are fitted ONLY on training data.
- No horizon conditioning is used in E2.
- Missing values are NOT imputed.
- Validation is used for model selection.
- Test set is used only for final evaluation.
"""

import os
import random

import numpy as np
import tensorflow as tf


from .config.config import ExperimentConfig

from .causal.var import TopologyAwareVAR
from .causal.causal_units import build_causal_units
from .causal.local_pem import LocalPEM

from .data.loader import load_numpy_dataset

from .data.preprocessing import (
    fit_max_scaler,
    normalize_data,
    create_sequences,
)

from .models.a3tgcn import A3TGCNModel

from .training.trainer import A3TGCNTrainer
from .training.scheduler import ValidationLRScheduler
from .training.evaluation import evaluate_model


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    data,
    train_rate=0.8,
    validation_rate=0.1,
):
    """
    Chronologically split traffic data.

    No shuffling is performed.

    Parameters
    ----------
    data:
        Traffic data [time, nodes]

    train_rate:
        Fraction used for training.

    validation_rate:
        Fraction used for validation.

    Returns
    -------
    train_data
    validation_data
    test_data
    """

    if data.ndim != 2:
        raise ValueError(
            f"Expected data with shape [time, nodes], got {data.shape}"
        )

    if not 0 < train_rate < 1:
        raise ValueError(
            f"train_rate must be between 0 and 1, got {train_rate}"
        )

    if not 0 <= validation_rate < 1:
        raise ValueError(
            f"validation_rate must be between 0 and 1, "
            f"got {validation_rate}"
        )

    if train_rate + validation_rate >= 1:
        raise ValueError(
            "train_rate + validation_rate must be less than 1."
        )

    num_timesteps = len(data)

    train_end = int(
        num_timesteps * train_rate
    )

    validation_end = int(
        num_timesteps *
        (train_rate + validation_rate)
    )

    train_data = data[:train_end]

    validation_data = data[
        train_end:validation_end
    ]

    test_data = data[
        validation_end:
    ]

    return (
        train_data,
        validation_data,
        test_data,
    )


# ============================================================
# VALID SEQUENCE FILTER
# ============================================================

def filter_valid_sequences(
    X,
    y,
):
    """
    Remove sequences containing NaN or Inf.

    No imputation is performed.

    A sequence is retained only when every value in both
    the input and target is finite.
    """

    if X.ndim != 3:
        raise ValueError(
            f"Expected X with shape [samples, seq_len, nodes], "
            f"got {X.shape}"
        )

    if y.ndim != 3:
        raise ValueError(
            f"Expected y with shape [samples, pre_len, nodes], "
            f"got {y.shape}"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    valid_X = np.all(
        np.isfinite(X),
        axis=(1, 2),
    )

    valid_y = np.all(
        np.isfinite(y),
        axis=(1, 2),
    )

    valid_mask = (
        valid_X
        & valid_y
    )

    return (
        X[valid_mask],
        y[valid_mask],
    )


# ============================================================
# BUILD E2 CAUSAL GRAPH
# ============================================================

def build_e2_causal_graph(
    train_data,
    adjacency,
):
    """
    Build the E2 probabilistic causal graph.

    Training data only:

        Training traffic
              ↓
        Topology-aware VAR
              ↓
        VAR residuals
              ↓
        Local PEM
              ↓
        P × D
              ↓
        Probabilistic directed causal graph
    """

    print()
    print("=" * 70)
    print("STEP 1: TOPOLOGY-AWARE VAR")
    print("=" * 70)

    print(
        "Training traffic shape:",
        train_data.shape,
    )

    print(
        "Adjacency shape:",
        adjacency.shape,
    )

    # --------------------------------------------------------
    # TOPOLOGY-AWARE VAR
    # --------------------------------------------------------

    var_model = TopologyAwareVAR(
        lag_order=1,
        ridge_alpha=1e-5,
        use_gpu=True,
    )

    var_result = var_model.fit(
        data=train_data,
        adjacency=adjacency,
    )

    print()
    print("VAR completed.")

    print(
        "VAR coefficients shape:",
        var_result.coefficients.shape,
    )

    print(
        "VAR residuals shape:",
        var_result.residuals.shape,
    )

    print(
        "VAR valid mask shape:",
        var_result.valid_mask.shape,
    )

    # ========================================================
    # CAUSAL UNITS
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 2: BUILD CAUSAL UNITS")
    print("=" * 70)

    units = build_causal_units(
        adjacency,
        include_self=True,
        max_unit_size=None,
        remove_self_loops=True,
    )

    unit_sizes = np.array(
        [
            len(unit.nodes)
            for unit in units
        ],
        dtype=np.int32,
    )

    print(
        "Number of causal units:",
        len(units),
    )

    print(
        "Minimum unit size:",
        unit_sizes.min(),
    )

    print(
        "Maximum unit size:",
        unit_sizes.max(),
    )

    print(
        "Mean unit size:",
        unit_sizes.mean(),
    )

    print(
        "Median unit size:",
        np.median(unit_sizes),
    )

    # ========================================================
    # RAW VS RESIDUAL RANK DIAGNOSTIC
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 3: RAW VS VAR-RESIDUAL RANK DIAGNOSTIC")
    print("=" * 70)

    diagnostic_unit = units[0]

    diagnostic_nodes = np.asarray(
        diagnostic_unit.nodes,
        dtype=np.int64,
    )

    print(
        "Diagnostic unit index:",
        0,
    )

    print(
        "Diagnostic target:",
        diagnostic_unit.target,
    )

    print(
        "Diagnostic nodes:",
        diagnostic_nodes.tolist(),
    )

    print(
        "Diagnostic unit size:",
        len(diagnostic_nodes),
    )

    # --------------------------------------------------------
    # RAW TRAFFIC
    # --------------------------------------------------------

    raw_local = train_data[
        :,
        diagnostic_nodes,
    ]

    raw_complete_mask = np.all(
        np.isfinite(raw_local),
        axis=1,
    )

    raw_local = raw_local[
        raw_complete_mask
    ]

    print(
        "Raw complete samples:",
        raw_local.shape[0],
    )

    if raw_local.shape[0] <= 1:
        raise RuntimeError(
            "Not enough complete raw samples "
            "for covariance diagnostic."
        )

    raw_centered = (
        raw_local
        - np.mean(
            raw_local,
            axis=0,
            keepdims=True,
        )
    )

    raw_covariance = (
        raw_centered.T
        @ raw_centered
        / raw_local.shape[0]
    )

    raw_rank = np.linalg.matrix_rank(
        raw_covariance
    )

    raw_eigenvalues = np.linalg.eigvalsh(
        raw_covariance
    )

    # --------------------------------------------------------
    # VAR RESIDUALS
    # --------------------------------------------------------

    residual_local = var_result.residuals[
        :,
        diagnostic_nodes,
    ]

    residual_complete_mask = np.all(
        np.isfinite(residual_local),
        axis=1,
    )

    residual_local = residual_local[
        residual_complete_mask
    ]

    print(
        "Residual complete samples:",
        residual_local.shape[0],
    )

    if residual_local.shape[0] <= 1:
        raise RuntimeError(
            "Not enough complete residual samples "
            "for covariance diagnostic."
        )

    residual_centered = (
        residual_local
        - np.mean(
            residual_local,
            axis=0,
            keepdims=True,
        )
    )

    residual_covariance = (
        residual_centered.T
        @ residual_centered
        / residual_local.shape[0]
    )

    residual_rank = np.linalg.matrix_rank(
        residual_covariance
    )

    residual_eigenvalues = np.linalg.eigvalsh(
        residual_covariance
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("RAW TRAFFIC COVARIANCE")
    print("-" * 70)

    print(
        "Variables:",
        raw_covariance.shape[0],
    )

    print(
        "Rank:",
        raw_rank,
    )

    print(
        "Expected rank:",
        raw_covariance.shape[0],
    )

    print(
        "Minimum eigenvalue:",
        raw_eigenvalues.min(),
    )

    print(
        "Maximum eigenvalue:",
        raw_eigenvalues.max(),
    )

    print()
    print("-" * 70)
    print("VAR RESIDUAL COVARIANCE")
    print("-" * 70)

    print(
        "Variables:",
        residual_covariance.shape[0],
    )

    print(
        "Rank:",
        residual_rank,
    )

    print(
        "Expected rank:",
        residual_covariance.shape[0],
    )

    print(
        "Minimum eigenvalue:",
        residual_eigenvalues.min(),
    )

    print(
        "Maximum eigenvalue:",
        residual_eigenvalues.max(),
    )

    print()
    print("-" * 70)
    print("RANK COMPARISON")
    print("-" * 70)

    print(
        "Raw traffic rank     :",
        raw_rank,
        "/",
        raw_covariance.shape[0],
    )

    print(
        "VAR residual rank    :",
        residual_rank,
        "/",
        residual_covariance.shape[0],
    )

    if raw_rank < raw_covariance.shape[0]:
        print()
        print(
            "WARNING: Raw traffic covariance "
            "is already rank deficient."
        )

    if residual_rank < residual_covariance.shape[0]:
        print()
        print(
            "WARNING: VAR residual covariance "
            "is rank deficient."
        )

    # ========================================================
    # LOCAL PEM
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 4: LOCAL PEM")
    print("=" * 70)

    pem_model = LocalPEM(
        units=units,
        nu0=0.05,
        nu1=1.0,
        tau=0.01,
        threshold=0.5,
        max_iter=500,
        tolerance=1e-6,
        spectral_norm_bound=None,
    )

    pem_result = pem_model.fit(
        residuals=var_result.residuals,
    )

    print()
    print("PEM completed.")

    print(
        "Processed units:",
        pem_result.processed_units,
    )

    print(
        "Global inclusion probabilities shape:",
        pem_result.global_inclusion_probabilities.shape,
    )

    print(
        "Global adjacency shape:",
        pem_result.global_adjacency.shape,
    )

    # ========================================================
    # BUILD PROBABILISTIC DIRECTED GRAPH
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 5: BUILD PROBABILISTIC CAUSAL GRAPH")
    print("=" * 70)

    inclusion_probabilities = (
        pem_result.global_inclusion_probabilities
    )

    directed_adjacency = (
        pem_result.global_adjacency
    )

    causal_graph = (
        inclusion_probabilities
        * directed_adjacency
    )

    np.fill_diagonal(
        causal_graph,
        0.0,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if causal_graph.shape != adjacency.shape:
        raise ValueError(
            "Causal graph shape mismatch: "
            f"{causal_graph.shape} != "
            f"{adjacency.shape}"
        )

    if not np.all(
        np.isfinite(causal_graph)
    ):
        raise ValueError(
            "Causal graph contains non-finite values."
        )

    print(
        "Causal graph shape:",
        causal_graph.shape,
    )

    print(
        "Causal graph nonzero edges:",
        np.count_nonzero(causal_graph),
    )

    print(
        "Causal graph min:",
        causal_graph.min(),
    )

    print(
        "Causal graph max:",
        causal_graph.max(),
    )

    print(
        "Causal graph mean:",
        causal_graph.mean(),
    )

    print()
    print("=" * 70)
    print("E2 CAUSAL GRAPH READY")
    print("=" * 70)

    return causal_graph


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("E2 - A3T-GCN + TOPOLOGY-AWARE VAR + PEM")
    print("=" * 70)

    # ========================================================
    # CONFIGURATION
    # ========================================================

    config = ExperimentConfig()

    set_seed(
        config.seed
    )

    print()
    print("Configuration")
    print("-" * 70)

    print(
        "Seed:",
        config.seed,
    )

    print(
        "GRU units:",
        config.gru_units,
    )

    print(
        "Sequence length:",
        config.seq_len,
    )

    print(
        "Prediction length:",
        config.pre_len,
    )

    print(
        "Learning rate:",
        config.learning_rate,
    )

    print(
        "Training epochs:",
        config.training_epoch,
    )

    print(
        "Batch size:",
        config.batch_size,
    )

    print(
        "Train rate:",
        config.train_rate,
    )

    print(
        "Lambda loss:",
        config.lambda_loss,
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    traffic_data, adjacency = load_numpy_dataset(
        config.data_path,
        config.adjacency_path,
    )

    traffic_data = np.asarray(
        traffic_data,
        dtype=np.float32,
    )

    adjacency = np.asarray(
        adjacency,
        dtype=np.float32,
    )

    print(
        "Traffic shape:",
        traffic_data.shape,
    )

    print(
        "Adjacency shape:",
        adjacency.shape,
    )

    print(
        "Traffic NaNs:",
        np.isnan(traffic_data).sum(),
    )

    print(
        "Traffic Inf:",
        np.isinf(traffic_data).sum(),
    )

    print(
        "Adjacency nonzero:",
        np.count_nonzero(adjacency),
    )

    # ========================================================
    # VALIDATE DATA
    # ========================================================

    if traffic_data.ndim != 2:
        raise ValueError(
            "Traffic data must have shape (T, N)."
        )

    if adjacency.ndim != 2:
        raise ValueError(
            "Adjacency must have shape (N, N)."
        )

    num_timesteps, num_nodes = (
        traffic_data.shape
    )

    if adjacency.shape != (
        num_nodes,
        num_nodes,
    ):
        raise ValueError(
            "Adjacency shape does not match "
            "number of traffic sensors."
        )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    train_data, val_data, test_data = (
        chronological_split(
            traffic_data,
            train_rate=0.8,
            validation_rate=0.1,
        )
    )

    print(
        "Train:",
        train_data.shape,
    )

    print(
        "Validation:",
        val_data.shape,
    )

    print(
        "Test:",
        test_data.shape,
    )

    # ========================================================
    # E2 CAUSAL GRAPH
    # ========================================================

    causal_graph = build_e2_causal_graph(
        train_data=train_data,
        adjacency=adjacency,
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print()
    print("=" * 70)
    print("NORMALIZATION")
    print("=" * 70)

    max_value = fit_max_scaler(
        train_data
    )

    print(
        "Training maximum:",
        max_value,
    )

    train_normalized = normalize_data(
        train_data,
        max_value,
    )

    val_normalized = normalize_data(
        val_data,
        max_value,
    )

    test_normalized = normalize_data(
        test_data,
        max_value,
    )

    print(
        "Train normalized:",
        train_normalized.shape,
    )

    print(
        "Validation normalized:",
        val_normalized.shape,
    )

    print(
        "Test normalized:",
        test_normalized.shape,
    )

    # ========================================================
    # CREATE SEQUENCES
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING SEQUENCES")
    print("=" * 70)

    train_x, train_y = create_sequences(
        train_normalized,
        seq_len=config.seq_len,
        pre_len=config.pre_len,
    )

    val_x, val_y = create_sequences(
        val_normalized,
        seq_len=config.seq_len,
        pre_len=config.pre_len,
    )

    test_x, test_y = create_sequences(
        test_normalized,
        seq_len=config.seq_len,
        pre_len=config.pre_len,
    )

    print(
        "Train sequences before filtering:",
        train_x.shape,
    )

    print(
        "Validation sequences before filtering:",
        val_x.shape,
    )

    print(
        "Test sequences before filtering:",
        test_x.shape,
    )

    # ========================================================
    # FILTER INVALID SEQUENCES
    # ========================================================

    train_x, train_y = (
        filter_valid_sequences(
            train_x,
            train_y,
        )
    )

    val_x, val_y = (
        filter_valid_sequences(
            val_x,
            val_y,
        )
    )

    test_x, test_y = (
        filter_valid_sequences(
            test_x,
            test_y,
        )
    )

    print()
    print(
        "Train sequences after filtering:",
        train_x.shape,
    )

    print(
        "Validation sequences after filtering:",
        val_x.shape,
    )

    print(
        "Test sequences after filtering:",
        test_x.shape,
    )

    # ========================================================
    # BUILD A3T-GCN
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING A3T-GCN")
    print("=" * 70)

    model = A3TGCNModel(
        seq_len=config.seq_len,
        num_nodes=num_nodes,
        gru_units=config.gru_units,
        pre_len=config.pre_len,
        adjacency=causal_graph,
        lambda_loss=config.lambda_loss,
    )

    print(
        "A3T-GCN model created."
    )

    print(
        "Graph supplied to A3T-GCN:",
        causal_graph.shape,
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = A3TGCNTrainer(
        model=model,
        learning_rate=config.learning_rate,
    )

    # ========================================================
    # VALIDATION SCHEDULER
    # ========================================================

    scheduler = ValidationLRScheduler(
        patience=5,
        factor=0.5,
        min_lr=1e-6,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING E2 A3T-GCN")
    print("=" * 70)

    checkpoint_dir = os.path.join(
        config.results_dir,
        "e2_checkpoints",
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    best_val_rmse = np.inf
    best_epoch = -1

    for epoch in range(
        config.training_epoch
    ):

        print()
        print(
            f"Epoch {epoch + 1}/"
            f"{config.training_epoch}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss = trainer.train_epoch(
            train_x,
            train_y,
            batch_size=config.batch_size,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_metrics = evaluate_model(
            trainer,
            val_x,
            val_y,
            max_value,
            batch_size=config.batch_size,
        )

        val_rmse = val_metrics["rmse"]

        print(
            "Train loss:",
            train_loss,
        )

        print(
            "Validation RMSE:",
            val_metrics["rmse"],
        )

        print(
            "Validation MAE:",
            val_metrics["mae"],
        )

        print(
            "Learning rate:",
            trainer.get_learning_rate(),
        )

        # ----------------------------------------------------
        # SAVE BEST VALIDATION MODEL
        # ----------------------------------------------------

        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse

            best_epoch = (
                epoch + 1
            )

            trainer.save_checkpoint(
                checkpoint_dir
            )

            print(
                "Best validation model saved."
            )

        # ----------------------------------------------------
        # LR SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_rmse,
            trainer,
        )

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("RESTORING BEST MODEL")
    print("=" * 70)

    trainer.restore_checkpoint(
        checkpoint_dir
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation RMSE:",
        best_val_rmse,
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_metrics = evaluate_model(
        trainer,
        test_x,
        test_y,
        max_value,
        batch_size=config.batch_size,
    )

    print()
    print("=" * 70)
    print("E2 FINAL RESULTS")
    print("=" * 70)

    print(
        "RMSE:",
        test_metrics["rmse"],
    )

    print(
        "MAE:",
        test_metrics["mae"],
    )

    print(
        "R2:",
        test_metrics["r2"],
    )

    print(
        "Accuracy:",
        test_metrics["accuracy"],
    )

    print()
    print("=" * 70)
    print("E2 COMPLETE")
    print("=" * 70)


# ============================================================
# DEBUG ENTRY POINT
# ============================================================

print("DEBUG: main.py loaded")

if __name__ == "__main__":
    print("DEBUG: calling main()")
    main()