"""
Kedro Pipeline Package.

Enthält die Pipeline-Definition und Nodes für die Breast Cancer Classification.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
