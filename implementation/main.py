import os
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from .data.preprocessing import (
    fit_max_scaler,
    normalize_data,
    create_sequences,
)

from .models.a3tgcn import A3TGCNModel

from .training.trainer import A3TGCNTrainer
from .training.scheduler import ValidationLRScheduler
from .training.metrics import (
    rmse,
    mae,
    r2_score,
    accuracy,
)

from .causal.var import TopologyAwareVAR
from .causal.causal_units import build_causal_units
from .causal.local_pem import LocalPEM


# ============================================================
# Configuration
# ============================================================

SEED = 42

SEQ_LEN = 12
PRE_LEN = 1

GRU_UNITS = 100

LEARNING_RATE = 0.005
TRAINING_EPOCH = 20
BATCH_SIZE = 8

TRAIN_RATE = 0.8
VAL_RATE = 0.1
TEST_RATE = 0.1

LAMBDA_LOSS = 0.0015

# VAR
VAR_LAG_ORDER = 1
VAR_RIDGE_ALPHA = 1e-5

# PEM
PEM_THRESHOLD = 0.5

# LR scheduler
LR_PATIENCE = 5
LR_FACTOR = 0.5
LR_MIN = 1e-5

# Checkpoint
CHECKPOINT_DIR = "results/e2_checkpoints"


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
tf.set_random_seed(SEED)


# ============================================================
# Data utilities
# ============================================================

def chronological_split(data):
    """
    Chronological 80/10/10 split.

    No shuffling is performed.
    """

    n = len(data)

    train_end = int(n * TRAIN_RATE)
    val_end = int(n * (TRAIN_RATE + VAL_RATE))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def filter_valid_sequences(X, y):
    """
    Remove sequences containing non-finite values.

    No imputation is performed.
    """

    X_valid = np.all(np.isfinite(X), axis=(1, 2))
    y_valid = np.all(np.isfinite(y), axis=(1, 2))

    valid_mask = X_valid & y_valid

    return X[valid_mask], y[valid_mask]


# ============================================================
# Data loading
# ============================================================

def load_dataset():

    traffic_path = os.path.join(
        os.path.dirname(__file__),
        "datasets",
        "LargeST GBA",
        "gba_his_raw_2021.h5",
    )

    adjacency_path = os.path.join(
        os.path.dirname(__file__),
        "datasets",
        "LargeST GBA",
        "gba_rn_adj.npy",
    )

    print("=" * 70)
    print("E2: VAR + PEM + A3T-GCN")
    print("=" * 70)

    print("\nLoading traffic data...")

    import h5py

    with h5py.File(traffic_path, "r") as f:
        traffic = f["traffic"][:]

    adjacency = np.load(adjacency_path)

    print(f"Traffic shape : {traffic.shape}")
    print(f"Adjacency shape : {adjacency.shape}")
    print(f"Traffic dtype : {traffic.dtype}")
    print(f"Adjacency dtype : {adjacency.dtype}")

    return traffic, adjacency


# ============================================================
# E2 causal graph
# ============================================================

def build_e2_causal_graph(train_data, adjacency):

    print("\n")
    print("=" * 70)
    print("E2 CAUSAL GRAPH")
    print("=" * 70)

    print("\nStep 1: Topology-aware VAR")

    var_model = TopologyAwareVAR(
        lag_order=VAR_LAG_ORDER,
        ridge_alpha=VAR_RIDGE_ALPHA,
        use_gpu=True,
    )

    var_result = var_model.fit(
        train_data,
        adjacency,
    )

    print("\nVAR completed.")

    print(
        "Coefficient shape :",
        var_result.coefficients.shape,
    )

    print(
        "Residual shape :",
        var_result.residuals.shape,
    )

    print(
        "Valid residuals :",
        np.sum(var_result.valid_mask),
    )

    # --------------------------------------------------------
    # Step 2: Build topology-aware causal units
    # --------------------------------------------------------

    print("\nStep 2: Building causal units")

    units = build_causal_units(
        adjacency
    )

    unit_sizes = np.array(
        [len(unit.nodes) for unit in units],
        dtype=np.int32,
    )

    print(
        f"Number of causal units : {len(units)}"
    )

    print(
        f"Unit size min           : {unit_sizes.min()}"
    )

    print(
        f"Unit size median        : {np.median(unit_sizes):.1f}"
    )

    print(
        f"Unit size mean          : {unit_sizes.mean():.2f}"
    )

    print(
        f"Unit size max           : {unit_sizes.max()}"
    )

    # --------------------------------------------------------
    # Step 3: Local PEM
    # --------------------------------------------------------

    print("\nStep 3: Local PEM")

    pem_model = LocalPEM(
        units=units,
        threshold=PEM_THRESHOLD,
    )

    pem_result = pem_model.fit(
        var_result.residuals
    )

    print("\nPEM completed.")

    print(
        "Processed units :",
        pem_result.processed_units,
    )

    # --------------------------------------------------------
    # Step 4: Construct probabilistic directed graph
    # --------------------------------------------------------
    #
    # global_inclusion_probabilities:
    #
    #     symmetric pairwise inclusion probability
    #
    # global_adjacency:
    #
    #     directed PEM-LBN adjacency
    #
    # Convention:
    #
    #     adjacency[target, source] = 1
    #
    # Therefore:
    #
    #     source ---> target
    #
    # Multiplying the two gives a directed graph whose
    # magnitude represents PEM inclusion probability.
    # --------------------------------------------------------

    print("\nStep 4: Constructing probabilistic causal graph")

    probability_matrix = (
        pem_result.global_inclusion_probabilities
    )

    directed_matrix = (
        pem_result.global_adjacency
    )

    causal_graph = (
        probability_matrix * directed_matrix
    )

    np.fill_diagonal(
        causal_graph,
        0.0,
    )

    causal_graph = np.asarray(
        causal_graph,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        "Causal graph shape :",
        causal_graph.shape,
    )

    print(
        "Finite graph :",
        np.all(np.isfinite(causal_graph)),
    )

    print(
        "Non-zero causal edges :",
        np.count_nonzero(causal_graph),
    )

    print(
        "Minimum edge weight :",
        causal_graph.min(),
    )

    print(
        "Maximum edge weight :",
        causal_graph.max(),
    )

    return causal_graph


# ============================================================
# Batch generation
# ============================================================

def iterate_batches(X, y, batch_size):

    n = len(X)

    for start in range(
        0,
        n,
        batch_size,
    ):

        end = min(
            start + batch_size,
            n,
        )

        yield (
            X[start:end],
            y[start:end],
        )


# ============================================================
# Prediction
# ============================================================

def predict_dataset(
    trainer,
    model,
    X,
    y,
):

    predictions = []
    targets = []

    for X_batch, y_batch in iterate_batches(
        X,
        y,
        BATCH_SIZE,
    ):

        feed_dict = {
            model.inputs: X_batch,
            model.labels: y_batch,
        }

        pred = trainer.session.run(
            model.y_pred,
            feed_dict=feed_dict,
        )

        predictions.append(pred)
        targets.append(
            y_batch.reshape(
                -1,
                model.num_nodes,
            )
        )

    if not predictions:

        return (
            np.empty(
                (0, model.num_nodes),
                dtype=np.float32,
            ),
            np.empty(
                (0, model.num_nodes),
                dtype=np.float32,
            ),
        )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    targets = np.concatenate(
        targets,
        axis=0,
    )

    return targets, predictions


# ============================================================
# Main
# ============================================================

def main():

    print("DEBUG: main.py loaded")
    print("DEBUG: starting E2")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    traffic, road_adjacency = load_dataset()

    num_nodes = traffic.shape[1]

    print(
        f"\nNumber of nodes : {num_nodes}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    print("\nSplitting data chronologically...")

    train_data, val_data, test_data = (
        chronological_split(traffic)
    )

    print(
        "Train shape :",
        train_data.shape,
    )

    print(
        "Validation shape :",
        val_data.shape,
    )

    print(
        "Test shape :",
        test_data.shape,
    )

    # --------------------------------------------------------
    # E2 causal graph
    #
    # IMPORTANT:
    # VAR + PEM use TRAIN DATA ONLY.
    # --------------------------------------------------------

    causal_graph = build_e2_causal_graph(
        train_data,
        road_adjacency,
    )

    # --------------------------------------------------------
    # Fit scaler on TRAIN data only
    # --------------------------------------------------------

    print("\nFitting scaler on training data...")

    max_value = fit_max_scaler(
        train_data
    )

    print(
        "Training max value :",
        max_value,
    )

    # --------------------------------------------------------
    # Normalize all splits using training scaler
    # --------------------------------------------------------

    train_norm = normalize_data(
        train_data,
        max_value,
    )

    val_norm = normalize_data(
        val_data,
        max_value,
    )

    test_norm = normalize_data(
        test_data,
        max_value,
    )

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    print("\nCreating training sequences...")

    X_train, y_train = create_sequences(
        train_norm,
        SEQ_LEN,
        PRE_LEN,
    )

    print(
        "Raw train sequences:",
        X_train.shape,
        y_train.shape,
    )

    print("\nCreating validation sequences...")

    X_val, y_val = create_sequences(
        val_norm,
        SEQ_LEN,
        PRE_LEN,
    )

    print(
        "Raw validation sequences:",
        X_val.shape,
        y_val.shape,
    )

    print("\nCreating test sequences...")

    X_test, y_test = create_sequences(
        test_norm,
        SEQ_LEN,
        PRE_LEN,
    )

    print(
        "Raw test sequences:",
        X_test.shape,
        y_test.shape,
    )

    # --------------------------------------------------------
    # Remove sequences containing NaN / Inf
    # --------------------------------------------------------

    X_train, y_train = filter_valid_sequences(
        X_train,
        y_train,
    )

    X_val, y_val = filter_valid_sequences(
        X_val,
        y_val,
    )

    X_test, y_test = filter_valid_sequences(
        X_test,
        y_test,
    )

    print("\nValid sequences:")

    print(
        "Train:",
        X_train.shape,
        y_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
        y_val.shape,
    )

    print(
        "Test:",
        X_test.shape,
        y_test.shape,
    )

    # --------------------------------------------------------
    # TensorFlow graph
    # --------------------------------------------------------

    tf.reset_default_graph()

    # Re-establish reproducibility after graph reset.
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    print("\nBuilding A3T-GCN model...")

    model = A3TGCNModel(
        seq_len=SEQ_LEN,
        num_nodes=num_nodes,
        gru_units=GRU_UNITS,
        pre_len=PRE_LEN,
        adjacency=causal_graph,
        lambda_loss=LAMBDA_LOSS,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = A3TGCNTrainer(
        loss=model.loss,
        learning_rate=LEARNING_RATE,
        optimizer="adam",
    )

    trainer.initialize()

    # --------------------------------------------------------
    # Validation LR scheduler
    # --------------------------------------------------------

    scheduler = ValidationLRScheduler(
        initial_lr=LEARNING_RATE,
        patience=LR_PATIENCE,
        factor=LR_FACTOR,
        min_lr=LR_MIN,
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "best_model",
    )

    best_val_rmse = float("inf")
    best_epoch = -1

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("A3T-GCN TRAINING")
    print("=" * 70)

    for epoch in range(
        TRAINING_EPOCH
    ):

        batch_losses = []

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for X_batch, y_batch in iterate_batches(
            X_train,
            y_train,
            BATCH_SIZE,
        ):

            feed_dict = {
                model.inputs: X_batch,
                model.labels: y_batch,
            }

            batch_loss = trainer.train_batch(
                feed_dict
            )

            batch_losses.append(
                batch_loss
            )

        train_loss = (
            float(np.mean(batch_losses))
            if batch_losses
            else float("nan")
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_targets, val_predictions = predict_dataset(
            trainer,
            model,
            X_val,
            y_val,
        )

        val_rmse = rmse(
            val_targets,
            val_predictions,
        )

        val_mae = mae(
            val_targets,
            val_predictions,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        current_lr = scheduler.step(
            val_rmse
        )

        trainer.set_learning_rate(
            current_lr
        )

        # ----------------------------------------------------
        # Save BEST model using validation RMSE
        #
        # Test data is NOT used here.
        # ----------------------------------------------------

        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse
            best_epoch = epoch

            trainer.save(
                checkpoint_path,
            )

            checkpoint_status = " * BEST"

        else:

            checkpoint_status = ""

        print(
            f"Epoch {epoch + 1:03d}/{TRAINING_EPOCH} | "
            f"Loss {train_loss:.6f} | "
            f"Val RMSE {val_rmse:.6f} | "
            f"Val MAE {val_mae:.6f} | "
            f"LR {current_lr:.8f}"
            f"{checkpoint_status}"
        )

    # --------------------------------------------------------
    # Restore best validation checkpoint
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RESTORING BEST MODEL")
    print("=" * 70)

    print(
        "Best epoch :",
        best_epoch + 1,
    )

    print(
        "Best validation RMSE :",
        best_val_rmse,
    )

    trainer.restore(
        checkpoint_path
    )

    # --------------------------------------------------------
    # Final TEST evaluation
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_targets, test_predictions = predict_dataset(
        trainer,
        model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Inverse scaling
    # --------------------------------------------------------

    test_targets_original = (
        test_targets * max_value
    )

    test_predictions_original = (
        test_predictions * max_value
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    test_rmse = rmse(
        test_targets_original,
        test_predictions_original,
    )

    test_mae = mae(
        test_targets_original,
        test_predictions_original,
    )

    test_r2 = r2_score(
        test_targets_original,
        test_predictions_original,
    )

    test_accuracy = accuracy(
        test_targets_original,
        test_predictions_original,
    )

    print(
        f"RMSE     : {test_rmse:.6f}"
    )

    print(
        f"MAE      : {test_mae:.6f}"
    )

    print(
        f"R2       : {test_r2:.6f}"
    )

    print(
        f"Accuracy : {test_accuracy:.6f}"
    )

    # --------------------------------------------------------
    # Close TensorFlow session
    # --------------------------------------------------------

    trainer.close()

    print("\n")
    print("=" * 70)
    print("E2 COMPLETE")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    print("DEBUG: calling main()")

    main()