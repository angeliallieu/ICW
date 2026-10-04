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
    node_evaluate
)


def create_pipeline() -> Pipeline:
    """
    Erstellt die Breast Cancer Classification Pipeline.

    Returns:
    --------
    pipeline : kedro.pipeline.Pipeline
        Die definierte Pipeline mit allen Nodes und Dependencies.


    Pipeline.node() API:
      pipeline += node(
          func=<Funktion>,
          inputs=<Input Namen (vom Catalog oder anderen Nodes)>,
          outputs=<Output Namen (speichert im Catalog)>,
          name=<Eindeutiger Name>,
          tags=[<Optional Tags>]
      )

    Inputs können sein:
      - String: Namen im Data Catalog oder Output einer anderen Node
      - "parameters": Special case für config Parameter

    Outputs:
      - String oder List[String]: Output wird unter diesem Namen im Catalog gespeichert
    """

    pipeline = Pipeline(
        [
            # =================================================================
            # NODE 1: Load Data
            # =================================================================
            node(
                func=node_load_data,
                inputs=None,  # Keine Inputs, liest von sklearn
                outputs=["raw_breast_cancer", "y"],
                name="load_data_node",
                tags=["data_loading"]
            ),

            # =================================================================
            # NODE 2: Clean Data
            # =================================================================
            node(
                func=node_clean_data,
                inputs=["raw_breast_cancer", "y"],  # Nimmt Outputs von load_data
                outputs=["cleaned_data", "y"],  # Speichert bereinigte X und y
                name="clean_data_node",
                tags=["data_processing"]
            ),

            # =================================================================
            # NODE 3: Normalize Features
            # =================================================================
            node(
                func=node_normalize_features,
                inputs=["cleaned_data", "y"],  # Nimmt von clean_data Node
                outputs=["normalized_data", "y", "feature_scaler"],  # Speichert Scaler!
                name="normalize_features_node",
                tags=["data_processing"]
            ),

            # =================================================================
            # NODE 4: Preprocess Features (Feature Selection)
            # =================================================================
            node(
                func=node_preprocess_features,
                inputs=["normalized_data", "y", "parameters"],  # parameters from config
                outputs=["preprocessed_data", "y"],
                name="preprocess_features_node",
                tags=["feature_engineering"]
            ),

            # =================================================================
            # NODE 5: Train/Test Split
            # =================================================================
            node(
                func=node_train_test_split,
                inputs=["preprocessed_data", "y", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],  # 4 Outputs!
                name="train_test_split_node",
                tags=["feature_engineering"]
            ),

            # =================================================================
            # NODE 6: Train Model
            # =================================================================
            node(
                func=node_train_model,
                inputs=["X_train", "y_train", "parameters"],
                outputs="trained_model",  # Saved as joblib im Catalog
                name="train_model_node",
                tags=["modeling"]
            ),

            # =================================================================
            # NODE 7: Make Predictions
            # =================================================================
            node(
                func=node_predict,
                inputs=["trained_model", "X_test"],
                outputs="predictions",
                name="predict_node",
                tags=["modeling"]
            ),

            # =================================================================
            # NODE 8: Evaluate Model
            # =================================================================
            node(
                func=node_evaluate,
                inputs=["y_test", "predictions", "trained_model"],
                outputs=["metrics", "confusion_matrix"],
                name="evaluate_node",
                tags=["evaluation"]
            ),
        ]
    )

    return pipeline


# if __name__ == "__main__":
#     visualize_pipeline()
