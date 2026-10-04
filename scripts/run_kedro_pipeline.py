#!/usr/bin/env python3

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.pipelines.kedro_pipeline.nodes import (
    node_load_data,
    node_clean_data,
    node_normalize_features,
    node_preprocess_features,
    node_train_test_split,
    node_train_model,
    node_predict,
    node_evaluate
)


def load_parameters(config_path: str = "config/parameters.yml"):
    """Lade Konfigurationsparameter"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_kedro_pipeline():
    """Führe Kedro Pipeline aus (Node für Node)"""
    
    print("\n" + "="*80)
    print(" KEDRO PIPELINE EXECUTION")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Load configuration
        print("\n[1/8] Loading configuration...")
        parameters = load_parameters()
        print("  ✓ Configuration loaded")
        print(f"    - Model: {parameters['model']['algorithm']}")
        print(f"    - Features to select: {parameters['feature_engineering']['n_features_to_select']}")
        print(f"    - Test size: {parameters['feature_engineering']['test_size']}")
        
        # Node 1: Load Data
        print("\n[2/8] Loading data...")
        X, y = node_load_data()
        print(f"  ✓ Data loaded: X.shape={X.shape}, y.shape={y.shape}")
        
        # Node 2: Clean Data
        print("\n[3/8] Cleaning data...")
        X_clean, y = node_clean_data(X, y)
        print(f"  ✓ Data cleaned: X.shape={X_clean.shape}")
        
        # Node 3: Normalize Features
        print("\n[4/8] Normalizing features...")
        X_norm, y, scaler = node_normalize_features(X_clean, y)
        print(f"  ✓ Features normalized: X.shape={X_norm.shape}")
        
        # Node 4: Preprocess Features (Feature Selection)
        print("\n[5/8] Preprocessing features (SelectKBest)...")
        X_proc, y = node_preprocess_features(X_norm, y, parameters)
        print(f"  ✓ Features selected: X.shape={X_proc.shape}")
        
        # Node 5: Train/Test Split
        print("\n[6/8] Splitting train/test...")
        X_train, X_test, y_train, y_test = node_train_test_split(X_proc, y, parameters)
        print(f"  ✓ Split complete:")
        print(f"    - Train: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"    - Test:  X_test={X_test.shape}, y_test={y_test.shape}")
        
        # Node 6: Train Model
        print("\n[7/8] Training model...")
        model = node_train_model(X_train, y_train, parameters)
        print(f"  ✓ Model trained (RandomForest)")
        
        # Node 7: Make Predictions
        print("\n[8/8] Making predictions...")
        predictions = node_predict(model, X_test)
        print(f"  ✓ Predictions made: shape={predictions.shape}")
        
        # Node 8: Evaluate
        print("\n[9/8] Evaluating model...")
        metrics, confusion_matrix = node_evaluate(y_test, predictions, model)
        print(f"  ✓ Evaluation complete")
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"""
Accuracy:  {metrics['accuracy']:.4f}
F1-Score:  {metrics['f1']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
ROC-AUC:   {metrics['roc_auc']:.4f}

Runtime: {runtime:.2f}s
        """)
        
        print("="*80)
        print("✅ Kedro Pipeline completed successfully!")
        print("="*80 + "\n")
        
        return {
            'status': 'success',
            'runtime': runtime,
            'accuracy': metrics['accuracy'],
            'metrics': metrics
        }
        
    except Exception as e:
        runtime = time.time() - start_time
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("="*80 + "\n")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'runtime': runtime,
            'error': str(e)
        }


if __name__ == "__main__":
    result = run_kedro_pipeline()
    sys.exit(0 if result['status'] == 'success' else 1)
