"""
=============================================================================
KEDRO PIPELINE: Pipeline Definition (Workflow)
=============================================================================
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    node_load_data,
    node_clean_data,
    node_normalize_features,
    node_preprocess_features,
    node_train_test_split,
    node_train_model,
    node_predict,
    node_evaluate,
)


def create_pipeline() -> Pipeline:
    """Erstellt die Breast Cancer Classification Pipeline."""

    return Pipeline(
        [
            # =================================================================
            # NODE 1: Load Data
            # =================================================================
            node(
                func=node_load_data,
                inputs=None,
                outputs=["raw_breast_cancer", "raw_y"],
                name="load_data_node",
                tags=["data_loading"],
            ),

            # =================================================================
            # NODE 2: Clean Data
            # =================================================================
            node(
                func=node_clean_data,
                inputs=["raw_breast_cancer", "raw_y"],
                outputs=["cleaned_data", "cleaned_y"],
                name="clean_data_node",
                tags=["data_processing"],
            ),

            # =================================================================
            # NODE 3: Normalize Features (nur Features skaliert)
            # =================================================================
            node(
                func=node_normalize_features,
                inputs="cleaned_data",
                outputs=["normalized_data", "feature_scaler"],
                name="normalize_features_node",
                tags=["data_processing"],
            ),

            # =================================================================
            # NODE 4: Preprocess Features (Feature Selection)
            # =================================================================
            node(
                func=node_preprocess_features,
                inputs=["normalized_data", "cleaned_y", "parameters"],
                outputs="preprocessed_data",
                name="preprocess_features_node",
                tags=["feature_engineering"],
            ),

            # =================================================================
            # NODE 5: Train/Test Split
            # =================================================================
            node(
                func=node_train_test_split,
                inputs=["preprocessed_data", "cleaned_y", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="train_test_split_node",
                tags=["feature_engineering"],
            ),

            # =================================================================
            # NODE 6: Train Model
            # =================================================================
            node(
                func=node_train_model,
                inputs=["X_train", "y_train", "parameters"],
                outputs="trained_model",
                name="train_model_node",
                tags=["modeling"],
            ),

            # =================================================================
            # NODE 7: Make Predictions
            # =================================================================
            node(
                func=node_predict,
                inputs=["trained_model", "X_test"],
                outputs="predictions",
                name="predict_node",
                tags=["modeling"],
            ),

            # =================================================================
            # NODE 8: Evaluate Model
            # =================================================================
            node(
                func=node_evaluate,
                inputs=["y_test", "predictions", "trained_model"],
                outputs=["metrics", "confusion_matrix"],
                name="evaluate_node",
                tags=["evaluation"],
            ),
        ]
    )