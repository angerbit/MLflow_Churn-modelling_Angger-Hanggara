import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urlunparse

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PACKAGE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DATASET_PATH = PACKAGE_DIR
DEFAULT_TARGET_COLUMN = os.getenv('TARGET_COLUMN', 'Exited')

# ── DagsHub auth ─────────────────────────────────
_dagshub_user  = os.getenv('DAGSHUB_USERNAME')
_dagshub_token = os.getenv('DAGSHUB_TOKEN')
_dagshub_repo  = os.getenv('DAGSHUB_REPO', 'SMSML_angger_hanggara')
_dagshub_owner = os.getenv('DAGSHUB_OWNER', 'angerbit')

if _dagshub_user and _dagshub_token:
    # Mode CI: set tracking URI & credentials langsung, skip OAuth
    _tracking_uri = (
        os.getenv('MLFLOW_TRACKING_URI')
        or f'https://dagshub.com/{_dagshub_owner}/{_dagshub_repo}.mlflow'
    )
    os.environ['MLFLOW_TRACKING_URI']      = _tracking_uri
    os.environ['MLFLOW_TRACKING_USERNAME'] = _dagshub_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = _dagshub_token
    mlflow.set_tracking_uri(_tracking_uri)
    logger.info('DagsHub auth via token (CI mode). Tracking URI: %s', _tracking_uri)
else:
    # local mode
    import dagshub
    dagshub.init(repo_owner=_dagshub_owner, repo_name=_dagshub_repo, mlflow=True)
    logger.info('DagsHub auth via interactive OAuth (local mode)')
# ─────────────────────────────────────────────────────────────────────────────

# Function to get MLflow tracking URI from environment variables or use default
def get_tracking_uri() -> str:
    uri = (
        os.getenv('MLFLOW_TRACKING_URI')
        or os.getenv('DAGSHUB_TRACKING_URI')
        or 'http://127.0.0.1:5000'
    )
    if uri:
        uri = uri.strip().rstrip('/')
    if not os.getenv('MLFLOW_TRACKING_USERNAME'):
        username = os.getenv('MLFLOW_USERNAME') or os.getenv('DAGSHUB_USERNAME')
        token = os.getenv('MLFLOW_TOKEN') or os.getenv('DAGSHUB_TOKEN')
        if username and token and uri and uri.startswith(('http://', 'https://')):
            parsed = urlparse(uri)
            netloc = f"{username}:{token}@{parsed.netloc}"
            uri = urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    return uri

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train churn model with MLflow.')
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--target-column', type=str, default=None)
    parser.add_argument('--test-size', type=float, default=None)
    parser.add_argument('--random-state', type=int, default=None)
    parser.add_argument('--max-iter', type=int, default=None)
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--tracking-uri', type=str, default=None)
    parser.add_argument('--x-data-path', type=str, default=None)
    parser.add_argument('--y-data-path', type=str, default=None)
    return parser.parse_args()


def get_latest_model_version(model_name: str) -> str:
    """Query MLflow Model Registry dan print versi terbaru ke stdout.

    Dipanggil dari CI dengan:
        MODEL_VERSION=$(python modelling.py --get-model-version churn_model)
    """

    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.error("Model '%s' tidak ditemukan di registry %s", model_name, tracking_uri)
        sys.exit(1)
    latest = max(versions, key=lambda v: int(v.version))
    print(latest.version)
    return latest.version


def load_dataset(dataset_path: Path, target_col: str):
    if dataset_path.is_dir():
        x_path = dataset_path / 'X_preprocessed.csv'
        y_path = dataset_path / 'y_preprocessed.csv'
        csv_files = list(dataset_path.glob('*.csv'))

        if x_path.exists() and y_path.exists():
            logger.info('Loading dataset from folder: %s', dataset_path)
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            return X, y

        if len(csv_files) == 1:
            logger.info('Loading dataset from single CSV file inside folder: %s', csv_files[0])
            return load_dataset(csv_files[0], target_col)

        raise FileNotFoundError(
            f'Folder {dataset_path} must contain X_preprocessed.csv and y_preprocessed.csv, or one CSV file.'
        )

    if dataset_path.exists() and dataset_path.is_file():
        logger.info('Loading dataset from file: %s', dataset_path)
        df = pd.read_csv(dataset_path)
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in {dataset_path}. "
                f"Available columns: {', '.join(df.columns)}"
            )
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    raise FileNotFoundError(f'Dataset path not found: {dataset_path}')


def save_classification_report(y_true, y_pred, output_path: Path) -> None:
    report_text = classification_report(y_true, y_pred)
    output_path.write_text(report_text, encoding='utf-8')


def load_xy_files(x_path: Path, y_path: Path):
    if not x_path.exists():
        raise FileNotFoundError(f'X file not found: {x_path}')
    if not y_path.exists():
        raise FileNotFoundError(f'y file not found: {y_path}')

    logger.info('Loading X from %s and y from %s', x_path, y_path)
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    if y.shape[1] > 1:
        logger.warning('y file contains more than one column; using the first column only')
        y = y.iloc[:, [0]]

    return X, y


def ensure_1d(y):
    return np.asarray(y).ravel()


def save_classification_report(y_true, y_pred, output_path: Path) -> None:
    report_text = classification_report(ensure_1d(y_true), ensure_1d(y_pred))
    output_path.write_text(report_text, encoding='utf-8')


def save_confusion_matrix_plot(y_true, y_pred, output_path: Path) -> None:
    y_true = ensure_1d(y_true)
    y_pred = ensure_1d(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    labels = np.unique(np.concatenate([y_true, y_pred]))
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha='center',
                va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_roc_curve_plot(y_true, y_scores, output_path: Path) -> None:
    y_true = ensure_1d(y_true)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='darkgray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_metadata(metadata: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metadata, indent=4), encoding='utf-8')


def log_artifacts(paths):
    for path in paths:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path='artifacts')
        else:
            logger.warning('Artifact not found: %s', path)


def main(args=None):
    if args is None:
        args = parse_args()

    logger.info('Starting MLflow training script...')
    logger.info('Python version: %s', sys.version)

    dataset_path = Path(args.dataset_path or os.getenv('DATASET_PATH', DEFAULT_DATASET_PATH))
    target_col = args.target_column or os.getenv('TARGET_COLUMN', DEFAULT_TARGET_COLUMN)
    x_data_path = args.x_data_path or os.getenv('X_DATA_PATH')
    y_data_path = args.y_data_path or os.getenv('Y_DATA_PATH')
    tracking_uri = args.tracking_uri or get_tracking_uri()
    experiment_name = args.experiment_name or os.getenv('MLFLOW_EXPERIMENT_NAME', 'churn_modeling')
    test_size = args.test_size if args.test_size is not None else float(os.getenv('TEST_SIZE', 0.2))
    random_state = args.random_state if args.random_state is not None else int(os.getenv('RANDOM_STATE', 42))
    max_iter = args.max_iter if args.max_iter is not None else int(os.getenv('MAX_ITER', 1000))

    logger.info('MLflow tracking URI: %s', tracking_uri)
    logger.info('MLflow experiment name: %s', experiment_name)
    logger.info('Dataset path: %s', dataset_path)
    logger.info('Target column: %s', target_col)

    if x_data_path and y_data_path:
        X, y = load_xy_files(Path(x_data_path), Path(y_data_path))
        dataset_source = f"X={x_data_path}, Y={y_data_path}"
    else:
        logger.info('Loading dataset from DATASET_PATH or default folder')
        X, y = load_dataset(dataset_path, target_col)
        dataset_source = str(dataset_path)
    logger.info('Loaded dataset X shape: %s, y shape: %s', X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(y.shape) == 1 or y.shape[1] == 1 else None,
    )

    with mlflow.start_run(run_name=f'LinearSVC_{datetime.now():%Y%m%d_%H%M%S}'):
        params = {
            'model_type': 'LinearSVC',
            'max_iter': max_iter,
            'random_state': random_state,
            'test_size': test_size,
            'target_column': target_col,
            'dataset_path': dataset_source,
        }
        mlflow.log_params(params)

        model = SVC(
            C=20.0,
            kernel='rbf',
            gamma='auto',
        )
        model.fit(X_train, y_train.values.ravel())

        y_pred = model.predict(X_test)
        y_scores = model.decision_function(X_test)

        y_test_1d = ensure_1d(y_test)
        y_pred_1d = ensure_1d(y_pred)

        accuracy = accuracy_score(y_test_1d, y_pred_1d)
        precision = precision_score(y_test_1d, y_pred_1d, zero_division=0)
        recall = recall_score(y_test_1d, y_pred_1d, zero_division=0)
        f1 = f1_score(y_test_1d, y_pred_1d, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test_1d, y_pred_1d).ravel()

        # ROC AUC
        try:
            fpr, tpr, _ = roc_curve(y_test_1d, y_scores)
            roc_auc = float(auc(fpr, tpr))
        except Exception:
            roc_auc = None

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'training_set_size': int(X_train.shape[0]),
            'testing_set_size': int(X_test.shape[0]),
        }
        if roc_auc is not None:
            metrics['roc_auc'] = float(roc_auc)

        mlflow.log_metrics(metrics)

        logger.info('Training completed. Accuracy=%s, F1=%s', accuracy, f1)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = MODEL_DIR / f'run_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / 'svc_model.pkl'
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path='models')

        report_path = run_dir / 'classification_report.txt'
        save_classification_report(y_test, y_pred, report_path)

        cm_plot_path = run_dir / 'confusion_matrix.png'
        save_confusion_matrix_plot(y_test, y_pred, cm_plot_path)

        roc_plot_path = run_dir / 'roc_curve.png'
        save_roc_curve_plot(y_test, y_scores, roc_plot_path)

        metadata_path = run_dir / 'model_metadata.json'
        save_metadata(
            {
                'dataset_path': dataset_source,
                'target_column': target_col,
                'model_type': 'SVC',
                'trained_at': timestamp,
                'features': int(X.shape[1]),
                'samples': int(X.shape[0]),
                **metrics,
            },
            metadata_path,
        )

        log_artifacts([report_path, cm_plot_path, roc_plot_path, metadata_path])

        logger.info('Artifacts saved to %s', run_dir)
        logger.info('Model training run finished.')


if __name__ == '__main__':
    _args = parse_args()
    if _args.get_model_version:
        get_latest_model_version(_args.get_model_version)
    else:
        main(args=_args)