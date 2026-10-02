#!/usr/bin/env python3
"""
Vergleicht Vanilla, Kedro-Nodes und Prefect.
"""

import json
import subprocess
import sys
import tempfile
import time
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Result:
    name: str
    runtime: float
    y_true: list
    y_pred: list
    y_pred_proba: list


def _to_list(values):
    """Konvertiert Pandas-/NumPy-Werte in JSON-kompatible Listen."""
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()

    if hasattr(values, "tolist"):
        values = values.tolist()
    else:
        values = list(values)

    return [
        value.item() if hasattr(value, "item") else value
        for value in values
    ]


def run_vanilla_worker():
    import yaml

    from src.core.data_processing import (
        load_breast_cancer,
        clean_data,
        normalize_features,
    )
    from src.core.feature_engineering import (
        preprocess_features,
        train_test_split,
    )
    from src.core.modeling import train_model, predict

    with open(PROJECT_ROOT / "config/parameters.yml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    X, y = load_breast_cancer()
    X_clean = clean_data(X)
    X_norm, _ = normalize_features(X_clean, fit=True)

    feature_config = config["feature_engineering"]
    X_proc, _ = preprocess_features(
        X_norm,
        y,
        n_features=feature_config["n_features_to_select"],
        random_state=feature_config["random_state"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc,
        y,
        test_size=feature_config["test_size"],
        random_state=feature_config["random_state"],
    )

    model_config = config["model"]
    model = train_model(
        X_train,
        y_train,
        model_config["algorithm"],
        model_config.get("random_forest", {}),
    )

    y_pred = predict(model, X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        "y_true": _to_list(y_test),
        "y_pred": _to_list(y_pred),
        "y_pred_proba": _to_list(y_pred_proba),
    }


def run_kedro_worker():
    import yaml

    from src.pipelines.kedro_pipeline.nodes import (
        node_load_data,
        node_clean_data,
        node_normalize_features,
        node_preprocess_features,
        node_train_test_split,
        node_train_model,
        node_predict,
    )

    with open(PROJECT_ROOT / "config/parameters.yml", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)

    X, y = node_load_data()
    X_clean, y = node_clean_data(X, y)
    X_norm, y, _ = node_normalize_features(X_clean, y)
    X_proc, y = node_preprocess_features(X_norm, y, parameters)
    X_train, X_test, y_train, y_test = node_train_test_split(
        X_proc, y, parameters
    )
    model = node_train_model(X_train, y_train, parameters)
    predictions = node_predict(model, X_test)

    return {
        "y_true": _to_list(y_test),
        "y_pred": _to_list(predictions["y_pred"]),
        "y_pred_proba": _to_list(predictions["y_pred_proba"]),
    }


def run_prefect_worker():
    from src.pipelines.prefect_pipeline.flows import (
        breast_cancer_pipeline_flow,
    )

    result = breast_cancer_pipeline_flow()
    return result["comparison"]


def worker_main(pipeline_name: str, result_path: Path):
    workers = {
        "vanilla": run_vanilla_worker,
        "kedro": run_kedro_worker,
        "prefect": run_prefect_worker,
    }

    if pipeline_name not in workers:
        raise ValueError(f"Unbekannte Pipeline: {pipeline_name}")

    result = workers[pipeline_name]()

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(result, file)


def run_isolated(pipeline_name: str, display_name: str) -> Result:
    """Führt eine Pipeline in einem frischen Prozess aus und misst End-to-End."""
    with tempfile.TemporaryDirectory(prefix="pipeline-comparison-") as temp_dir:
        temp_path = Path(temp_dir)
        result_path = temp_path / "result.json"
        log_path = temp_path / "pipeline.log"

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            pipeline_name,
            str(result_path),
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        if pipeline_name == "prefect":
            # Keine eventuell konfigurierte externe Prefect-API verwenden.
            env.pop("PREFECT_API_URL", None)

            # Lokale Prefect-Kommunikation am Proxy vorbeileiten.
            no_proxy_hosts = {
                value.strip()
                for variable in ("NO_PROXY", "no_proxy")
                for value in env.get(variable, "").split(",")
                if value.strip()
            }
            no_proxy_hosts.update({"localhost", "127.0.0.1"})

            no_proxy_value = ",".join(sorted(no_proxy_hosts))
            env["NO_PROXY"] = no_proxy_value
            env["no_proxy"] = no_proxy_value

        start = time.perf_counter()

        with open(log_path, "w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                env=env,
            )

        runtime = time.perf_counter() - start

        if completed.returncode != 0:
            print(f"\n--- Ausgabe von {display_name} ---")
            if log_path.exists():
                print(log_path.read_text(encoding="utf-8", errors="replace"))
            raise RuntimeError(
                f"{display_name} ist mit Exit-Code "
                f"{completed.returncode} fehlgeschlagen."
            )

        if not result_path.exists():
            raise RuntimeError(
                f"{display_name} hat keine Ergebnisdatei erzeugt."
            )

        data = json.loads(result_path.read_text(encoding="utf-8"))

        return Result(
            name=display_name,
            runtime=runtime,
            y_true=data["y_true"],
            y_pred=data["y_pred"],
            y_pred_proba=data["y_pred_proba"],
        )

def calculate_metrics(result: Result) -> dict:
    """Berechnet für jede Pipeline dieselben Metriken."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(result.y_true)
    y_pred = np.asarray(result.y_pred)
    y_pred_proba = np.asarray(result.y_pred_proba)

    if not (len(y_true) == len(y_pred) == len(y_pred_proba)):
        raise ValueError(
            f"{result.name}: y_true, y_pred und y_pred_proba "
            "haben unterschiedliche Längen."
        )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        # Zeilen = tatsächliche Klasse, Spalten = vorhergesagte Klasse
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def compare_predictions(results: list[Result]):
    import numpy as np

    reference = results[0]
    reference_y_true = np.asarray(reference.y_true)
    reference_y_pred = np.asarray(reference.y_pred)
    reference_proba = np.asarray(reference.y_pred_proba)

    print("\nVORHERSAGEVERGLEICH (jeweils gegen Vanilla)")
    print("-" * 80)

    for result in results[1:]:
        y_true = np.asarray(result.y_true)

        same_test_set = np.array_equal(reference_y_true, y_true)

        if same_test_set:
            same_predictions = np.array_equal(
                reference_y_pred,
                np.asarray(result.y_pred),
            )
            same_probabilities = np.allclose(
                reference_proba,
                np.asarray(result.y_pred_proba),
                rtol=1e-8,
                atol=1e-10,
            )
        else:
            same_predictions = False
            same_probabilities = False

        print(
            f"{result.name}: "
            f"gleiche Testlabels={same_test_set}, "
            f"gleiche Klassen-Vorhersagen={same_predictions}, "
            f"gleiche Wahrscheinlichkeiten (Toleranz)={same_probabilities}"
        )


def comparison_main():
    print("\n" + "=" * 80)
    print("ML PIPELINE COMPARISON: Vanilla vs. Kedro vs. Prefect")
    print("Laufzeit: frischer Prozessstart bis Prozessende")
    print("Prefect-Serverstart und -stopp sind enthalten.")
    print("=" * 80)

    pipelines = [
        ("vanilla", "Vanilla"),
        ("kedro", "Kedro-Nodes"),
        ("prefect", "Prefect"),
    ]

    results = []

    for index, (pipeline_name, display_name) in enumerate(pipelines, start=1):
        print(f"\n[{index}/3] Running {display_name}...")

        try:
            result = run_isolated(pipeline_name, display_name)
            results.append(result)
            print(
                f"[{display_name.upper()}] ✓ Success | "
                f"End-to-end Runtime: {result.runtime:.2f}s"
            )
        except Exception as error:
            print(f"\n[{display_name.upper()}] ✗ Failed: {error}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("METRIKEN")
    print("Confusion Matrix: Zeilen = Actual [0, 1], Spalten = Predicted [0, 1]")
    print("=" * 80)

    for result in results:
        metrics = calculate_metrics(result)
        print(f"\n{result.name} | Laufzeit: {result.runtime:.2f}s")
        print(
            f"  Accuracy:  {metrics['accuracy']:.4f}\n"
            f"  Precision: {metrics['precision']:.4f}\n"
            f"  Recall:    {metrics['recall']:.4f}\n"
            f"  F1:        {metrics['f1']:.4f}\n"
            f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n"
            f"  Confusion Matrix: {metrics['confusion_matrix']}"
        )

    compare_predictions(results)

    print("\n" + "=" * 80)
    print("Alle drei Pipelines wurden erfolgreich ausgeführt.")
    print("=" * 80 + "\n")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], Path(sys.argv[3]))
    else:
        comparison_main()


if __name__ == "__main__":
    main()