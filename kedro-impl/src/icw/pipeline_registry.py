"""Project pipelines."""

import logging

from kedro.pipeline import Pipeline

logger = logging.getLogger(__name__)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""

    from icw.pipelines.data_science import (
        create_pipeline as create_data_science_pipeline,
    )
    data_science_pipeline = create_data_science_pipeline()


    return {
        "data_science": data_science_pipeline,
        "__default__": data_science_pipeline,
    }
