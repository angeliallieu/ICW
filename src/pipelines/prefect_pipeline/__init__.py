"""
Prefect Pipeline Package.

Enthält die Prefect Flow-Definition und Deployment-Konfiguration
für die Breast Cancer Classification.

Ausführung:
  - Local: python -m src.pipelines.prefect_pipeline.flows
"""

from .flows import breast_cancer_pipeline_flow

__all__ = ["breast_cancer_pipeline_flow"]