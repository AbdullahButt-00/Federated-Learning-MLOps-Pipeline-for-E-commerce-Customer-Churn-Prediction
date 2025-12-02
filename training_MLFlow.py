#!/usr/bin/env python
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import mlflow
import mlflow.tensorflow
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.utils import shuffle

# Optional: interactive plots
import plotly.graph_objects as go

# ===================== CONFIG =====================
DATA_FOLDER = "preprocessed_data"
OUTPUT_FOLDER = "federated_data"
MODEL_SAVE_PATH = os.path.join(OUTPUT_FOLDER, "federated_churn_model.h5")
BATCH_SIZE = 8
NUM_ROUNDS = 20             # change as needed
LEARNING_RATE_CLIENT = 0.01
LEARNING_RATE_SERVER = 1.0
TEST_FRAC = 0.2
RANDOM_STATE = 42

# smoothing window for moving average (odd preferred)
SMOOTH_WINDOW = 3  # set to 3; you can increase to smooth more

# MLflow model registry name
MODEL_REGISTRY_NAME = "churn_prediction_model"

# ===================== MLflow =====================
# Connect to local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("federated_churn_prediction")

# ===================== DATASET HELPERS =====================
def create_tf_dataset(X, y, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X, dtype=tf.float32),
                                             tf.convert_to_tensor(y, dtype=tf.int32)))
    return ds.batch(batch_size)

# ===================== MODEL DEFINITION =====================
def create_keras_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# ===================== MAIN =====================
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train federated churn prediction model with MLflow versioning')
    parser.add_argument(
        '--dataset',
        type=str,
        default='E_Commerce_Dataset.xlsx',
        help='Path to the Excel dataset file (default: E_Commerce_Dataset.xlsx)'
    )
    parser.add_argument(
        '--sheet',
        type=str,
        default='E Comm',
        help='Sheet name in the Excel file (default: E Comm)'
    )
    parser.add_argument(
        '--data-folder',
        type=str,
        default='preprocessed_data',
        help='Path to preprocessed data folder (default: preprocessed_data)'
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        default='federated_data',
        help='Output folder for trained model (default: federated_data)'
    )
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=20,
        help='Number of federated training rounds (default: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )
    parser.add_argument(
        '--test-frac',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Use arguments
    DATASET_PATH = args.dataset
    SHEET_NAME = args.sheet
    DATA_FOLDER = args.data_folder
    OUTPUT_FOLDER = args.output_folder
    NUM_ROUNDS = args.num_rounds
    BATCH_SIZE = args.batch_size
    TEST_FRAC = args.test_frac
    MODEL_SAVE_PATH = os.path.join(OUTPUT_FOLDER, "federated_churn_model.h5")
    
    print(f"Dataset: {DATASET_PATH}")
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Number of rounds: {NUM_ROUNDS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Test fraction: {TEST_FRAC}")
    
    # ----------------- Load federated client datasets -----------------
    client_files = sorted([
        os.path.join(DATA_FOLDER, f)
        for f in os.listdir(DATA_FOLDER)
        if f.startswith("client_") and f.endswith("_data.pkl")
    ])

    if not client_files:
        raise RuntimeError(f"No client files found in {DATA_FOLDER}. Run preprocess.py first.")

    federated_train = []
    for file in client_files:
        with open(file, "rb") as f:
            data = pickle.load(f)
            X, y = data['X'], data['y']
            ds = create_tf_dataset(X, y)
            federated_train.append(ds)

    print(f"Loaded {len(federated_train)} federated client datasets")

    # ----------------- Prepare held-out test set (fixed) -----------------
    with open(os.path.join(DATA_FOLDER, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)

    raw_df = pd.read_excel(DATASET_PATH, sheet_name=SHEET_NAME)
    raw_df['Churn'] = raw_df['Churn'].astype(int)
    raw_df = shuffle(raw_df, random_state=RANDOM_STATE)

    test_df = raw_df.sample(frac=TEST_FRAC, random_state=RANDOM_STATE)
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn'].values
    X_test_trans = preprocessor.transform(X_test).astype(np.float32)
    print(f"Prepared test split: {len(y_test)} rows")

    # ----------------- Determine input shape -----------------
    for batch in federated_train[0].take(1):
        example_input = batch[0]
        break
    input_shape = int(example_input.shape[1])

    def model_fn():
        keras_model = create_keras_model(input_shape)
        return tff.learning.models.from_keras_model(
            keras_model=keras_model,
            input_spec=federated_train[0].element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
        )

    # ===================== TRAINING & MLflow RUN =====================
    round_eval_dir = os.path.join(OUTPUT_FOLDER, "round_evaluation")
    os.makedirs(round_eval_dir, exist_ok=True)

    with mlflow.start_run(run_name=f"federated_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:

        # Log params
        mlflow.log_param("num_clients", len(federated_train))
        mlflow.log_param("num_rounds", NUM_ROUNDS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("client_lr", LEARNING_RATE_CLIENT)
        mlflow.log_param("server_lr", LEARNING_RATE_SERVER)
        mlflow.log_param("test_frac", TEST_FRAC)
        mlflow.log_param("model_architecture", "64-32-1")
        mlflow.log_param("smoothing_window", SMOOTH_WINDOW)
        mlflow.log_param("dataset_path", DATASET_PATH)
        mlflow.log_param("sheet_name", SHEET_NAME)

        # Prepare trainer
        trainer = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn=model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CLIENT),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER)
        )

        state = trainer.initialize()

        # containers for per-round evaluation
        rounds = []
        train_losses = []
        train_accuracies = []
        train_aucs = []

        eval_accuracies = []
        eval_precisions = []
        eval_recalls = []
        eval_f1s = []
        eval_roc_aucs = []

        for round_num in range(NUM_ROUNDS):
            # perform federated round
            state, metrics = trainer.next(state, federated_train)
            print(f"Round {round_num+1}, metrics: {metrics}")

            # log training-side metrics (from clients aggregate)
            train_metrics = metrics['client_work']['train']
            train_loss = float(train_metrics['loss'])
            train_acc = float(train_metrics['binary_accuracy'])
            train_auc = float(train_metrics['auc'])

            mlflow.log_metric("train_loss", train_loss, step=round_num)
            mlflow.log_metric("train_accuracy", train_acc, step=round_num)
            mlflow.log_metric("train_auc", train_auc, step=round_num)

            # keep in lists for plotting
            rounds.append(round_num + 1)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_aucs.append(train_auc)

            # ---------------- Evaluate the GLOBAL model (current state) ----------------
            # Build a fresh Keras model and assign global weights
            central_model = create_keras_model(input_shape)
            global_weights = state.global_model_weights

            # assign trainable and non-trainable
            for var, val in zip(central_model.trainable_variables, global_weights.trainable):
                var.assign(val)
            for var, val in zip(central_model.non_trainable_variables, global_weights.non_trainable):
                var.assign(val)

            # Predict on the held-out test set
            y_pred_prob = central_model.predict(X_test_trans, batch_size=BATCH_SIZE).flatten()
            y_pred = (y_pred_prob >= 0.5).astype(int)

            # Compute evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_prob)

            # Log evaluation metrics per round to MLflow (use step as round_num)
            mlflow.log_metric("eval_accuracy", acc, step=round_num)
            mlflow.log_metric("eval_precision", prec, step=round_num)
            mlflow.log_metric("eval_recall", rec, step=round_num)
            mlflow.log_metric("eval_f1", f1, step=round_num)
            mlflow.log_metric("eval_roc_auc", roc_auc, step=round_num)

            # keep for plotting
            eval_accuracies.append(acc)
            eval_precisions.append(prec)
            eval_recalls.append(rec)
            eval_f1s.append(f1)
            eval_roc_aucs.append(roc_auc)

            # Save per-round metrics row to CSV (append)
            row = {
                "round": round_num + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_auc": train_auc,
                "eval_accuracy": acc,
                "eval_precision": prec,
                "eval_recall": rec,
                "eval_f1": f1,
                "eval_roc_auc": roc_auc
            }
            df_row = pd.DataFrame([row])
            metrics_csv = os.path.join(round_eval_dir, "per_round_metrics.csv")
            if round_num == 0:
                df_row.to_csv(metrics_csv, index=False)
            else:
                df_row.to_csv(metrics_csv, mode='a', header=False, index=False)

        # ===================== After all rounds: Save final global model =====================
        central_model.save(MODEL_SAVE_PATH)
        print(f"Saved federated model -> {MODEL_SAVE_PATH}")
        
        # Log final metrics
        final_accuracy = eval_accuracies[-1]
        final_precision = eval_precisions[-1]
        final_recall = eval_recalls[-1]
        final_f1 = eval_f1s[-1]
        final_roc_auc = eval_roc_aucs[-1]
        
        mlflow.log_metric("final_accuracy", final_accuracy)
        mlflow.log_metric("final_precision", final_precision)
        mlflow.log_metric("final_recall", final_recall)
        mlflow.log_metric("final_f1", final_f1)
        mlflow.log_metric("final_roc_auc", final_roc_auc)
        
        # Log model to MLflow with model registry
        print("\nRegistering model to MLflow Model Registry...")
        model_info = mlflow.tensorflow.log_model(
            central_model, 
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME
        )
        
        # Get the registered model version
        client = mlflow.tracking.MlflowClient()
        
        import time
        time.sleep(1)
        
        versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
        model_version = None
        
        for mv in versions:
            if mv.run_id == run.info.run_id:
                model_version = mv.version
                break
        
        if model_version is None:
            model_version = max([int(mv.version) for mv in versions])
        
        print(f"✓ Model registered as '{MODEL_REGISTRY_NAME}' version {model_version}")
        
        # Add model version tag
        mlflow.set_tag("model_version", model_version)
        mlflow.set_tag("model_name", MODEL_REGISTRY_NAME)
        
        # Tag with performance metrics
        client.set_model_version_tag(
            name=MODEL_REGISTRY_NAME,
            version=model_version,
            key="accuracy",
            value=str(round(final_accuracy, 4))
        )
        client.set_model_version_tag(
            name=MODEL_REGISTRY_NAME,
            version=model_version,
            key="f1_score",
            value=str(round(final_f1, 4))
        )
        
        # ===================== MODEL PROMOTION / ROLLBACK LOGIC =====================
        MIN_ACCURACY_THRESHOLD = 0.80
        
        print("\n" + "="*60)
        print("EVALUATING MODEL VERSION FOR PROMOTION")
        print("="*60)
        
        # Check if this version meets minimum accuracy threshold
        if final_accuracy < MIN_ACCURACY_THRESHOLD:
            print(f"⚠️  Model accuracy ({final_accuracy:.4f}) is below threshold ({MIN_ACCURACY_THRESHOLD})")
            
            # Archive this version
            try:
                client.transition_model_version_stage(
                    name=MODEL_REGISTRY_NAME,
                    version=model_version,
                    stage="Archived",
                    archive_existing_versions=False
                )
                print(f"❌ Version {model_version} ARCHIVED (below accuracy threshold)")
            except Exception as e:
                print(f"Warning: Could not archive version: {e}")
            
            # Check if there's a previous Production version to keep
            production_versions = [v for v in versions if v.current_stage == "Production"]
            if production_versions:
                prod_version = production_versions[0].version
                print(f"✓ Keeping previous Production version {prod_version}")
            else:
                print("⚠️  No existing Production model - manual intervention may be required")
        
        else:
            # Model meets threshold, now compare with previous version
            previous_production = None
            previous_accuracy = None
            
            # Refresh versions list to get current stages
            versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
            
            # Get current production version if exists (excluding the version we just created)
            for v in versions:
                if v.current_stage == "Production" and v.version != model_version:
                    previous_production = v
                    # Get accuracy from tags
                    if 'accuracy' in v.tags:
                        previous_accuracy = float(v.tags['accuracy'])
                    break
            
            # If no Production version found, check if there's any previous version at all
            if not previous_production:
                # Get all versions except current one, sorted by version number
                previous_versions = sorted(
                    [v for v in versions if v.version != model_version],
                    key=lambda x: int(x.version),
                    reverse=True
                )
                if previous_versions:
                    previous_production = previous_versions[0]
                    if 'accuracy' in previous_production.tags:
                        previous_accuracy = float(previous_production.tags['accuracy'])
            
            # Compare with previous production version
            if previous_production and previous_accuracy:
                print(f"\nComparing with Production version {previous_production.version}:")
                print(f"  Current version {model_version}: accuracy = {final_accuracy:.4f}")
                print(f"  Previous version {previous_production.version}: accuracy = {previous_accuracy:.4f}")
                
                if final_accuracy < previous_accuracy:
                    print(f"\n⚠️  Current version performs WORSE than previous version")
                    print(f"🔄 ROLLBACK: Keeping version {previous_production.version} in Production")
                    
                    # Archive the new worse version
                    try:
                        client.transition_model_version_stage(
                            name=MODEL_REGISTRY_NAME,
                            version=model_version,
                            stage="Archived",
                            archive_existing_versions=False
                        )
                        print(f"❌ Version {model_version} ARCHIVED (performance regression)")
                    except Exception as e:
                        print(f"Warning: Could not archive version: {e}")
                    
                    # Keep previous version in production (it should already be there)
                    print(f"✓ Version {previous_production.version} remains in Production")
                    model_version = previous_production.version  # Update for final message
                    final_accuracy = previous_accuracy
                    
                else:
                    # New version is better, promote it
                    improvement = ((final_accuracy - previous_accuracy) / previous_accuracy) * 100
                    print(f"\n✓ Current version is BETTER (+{improvement:.2f}% improvement)")
                    print(f"⬆️  PROMOTING version {model_version} to Production")
                    
                    try:
                        client.transition_model_version_stage(
                            name=MODEL_REGISTRY_NAME,
                            version=model_version,
                            stage="Production",
                            archive_existing_versions=True  # Archive old production versions
                        )
                        print(f"✓ Version {model_version} promoted to Production")
                        print(f"✓ Previous version {previous_production.version} archived")
                    except Exception as e:
                        print(f"Warning: Could not promote version: {e}")
            
            else:
                # No previous production version, promote this one
                print(f"\nNo previous Production version found")
                print(f"⬆️  PROMOTING version {model_version} to Production")
                
                try:
                    client.transition_model_version_stage(
                        name=MODEL_REGISTRY_NAME,
                        version=model_version,
                        stage="Production",
                        archive_existing_versions=False
                    )
                    print(f"✓ Version {model_version} promoted to Production")
                except Exception as e:
                    print(f"Warning: Could not promote version: {e}")
        
        print("="*60)
        
        # Also log the model file as artifact
        mlflow.log_artifact(MODEL_SAVE_PATH)

        # ----------------- Create aesthetic matplotlib plots with smoothing + CI -----------------
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-darkgrid")

        def smooth_and_ci(series, window=SMOOTH_WINDOW):
            s = pd.Series(series)
            # moving average (centered)
            smooth = s.rolling(window=window, center=True, min_periods=1).mean()
            # rolling std
            std = s.rolling(window=window, center=True, min_periods=1).std().fillna(0.0)
            # 95% approx band using 1.96 * std (note: not true CI across runs, but indicates variability)
            upper = smooth + 1.96 * std
            lower = smooth - 1.96 * std
            return smooth.values, lower.values, upper.values

        def plot_with_ci(x, y, ylabel, title, outpath_png, outpath_svg):
            smooth, lower, upper = smooth_and_ci(y, window=SMOOTH_WINDOW)
            plt.figure(figsize=(9,5))
            plt.plot(x, y, marker='o', linestyle='-', alpha=0.35, label='raw', linewidth=1.5)
            plt.plot(x, smooth, marker='o', linestyle='-', linewidth=2.2, label='smoothed')
            plt.fill_between(x, lower, upper, color='gray', alpha=0.25, label='±1.96·std (rolling)')
            plt.xticks(x)
            plt.xlabel("Round")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outpath_png, dpi=160)
            plt.savefig(outpath_svg)
            plt.close()

        # craft metric plots
        plot_with_ci(rounds, eval_accuracies, "Accuracy", "Eval Accuracy vs Round (smoothed)", os.path.join(round_eval_dir, "eval_accuracy_vs_round.png"), os.path.join(round_eval_dir, "eval_accuracy_vs_round.svg"))
        plot_with_ci(rounds, eval_precisions, "Precision", "Eval Precision vs Round (smoothed)", os.path.join(round_eval_dir, "eval_precision_vs_round.png"), os.path.join(round_eval_dir, "eval_precision_vs_round.svg"))
        plot_with_ci(rounds, eval_recalls, "Recall", "Eval Recall vs Round (smoothed)", os.path.join(round_eval_dir, "eval_recall_vs_round.png"), os.path.join(round_eval_dir, "eval_recall_vs_round.svg"))
        plot_with_ci(rounds, eval_f1s, "F1-score", "Eval F1-score vs Round (smoothed)", os.path.join(round_eval_dir, "eval_f1_vs_round.png"), os.path.join(round_eval_dir, "eval_f1_vs_round.svg"))
        plot_with_ci(rounds, eval_roc_aucs, "ROC AUC", "Eval ROC AUC vs Round (smoothed)", os.path.join(round_eval_dir, "eval_roc_auc_vs_round.png"), os.path.join(round_eval_dir, "eval_roc_auc_vs_round.svg"))

        # ----------------- Create interactive Plotly chart with raw + smoothed + CI -----------------
        def make_plotly_combined(rounds, metric_dict, out_html):
            """
            metric_dict: dict of name -> list(values)
            Creates an interactive chart with traces for raw and smoothed series plus shaded CI.
            """
            fig = go.Figure()
            for name, values in metric_dict.items():
                s = pd.Series(values)
                smooth = s.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
                std = s.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).std().fillna(0.0)
                upper = (smooth + 1.96 * std).values
                lower = (smooth - 1.96 * std).values

                # raw trace
                fig.add_trace(go.Scatter(x=rounds, y=values, mode='markers+lines', name=f"{name} (raw)", opacity=0.4))
                # smoothed trace
                fig.add_trace(go.Scatter(x=rounds, y=smooth, mode='lines+markers', name=f"{name} (smoothed)", line=dict(width=3)))
                # CI band (fill between lower and upper)
                fig.add_trace(go.Scatter(
                    x=rounds + rounds[::-1],
                    y=list(upper) + list(lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(150,150,150,0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{name} CI"
                ))

            fig.update_layout(
                title="Per-round Evaluation Metrics (raw + smoothed + CI)",
                xaxis_title="Round",
                yaxis_title="Metric value",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.write_html(out_html, include_plotlyjs='cdn')

        metrics_for_plotly = {
            "Accuracy": eval_accuracies,
            "Precision": eval_precisions,
            "Recall": eval_recalls,
            "F1": eval_f1s,
            "ROC AUC": eval_roc_aucs
        }
        plotly_html = os.path.join(round_eval_dir, "per_round_metrics_interactive.html")
        make_plotly_combined(rounds, metrics_for_plotly, plotly_html)

        # ---------------- Save and log artifacts to MLflow -----------------
        # Save per-round CSV has been written during loop
        mlflow.log_artifact(metrics_csv, artifact_path="round_evaluation")

        # Log all plot files in round_eval_dir
        for fn in os.listdir(round_eval_dir):
            mlflow.log_artifact(os.path.join(round_eval_dir, fn), artifact_path="round_evaluation")

        print("\nPer-round evaluation metrics, plots (matplotlib & plotly) saved to:", round_eval_dir)
        print("They have also been logged to MLflow under artifact path 'round_evaluation'.")

        # ===================== END RUN =====================
        print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Run ID: {run.info.run_id}")
        print(f"\n{'='*60}")
        print(f"✓ Training complete!")
        print(f"  Model: {MODEL_REGISTRY_NAME} v{model_version}")
        print(f"  Final Accuracy: {final_accuracy:.4f}")
        print(f"  Final F1 Score: {final_f1:.4f}")
        print(f"{'='*60}\n")