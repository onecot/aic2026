"""QA task foundation built on the shared retrieval architecture.

Modules:
    - :mod:`aic2026.qa.qa_types`: typed QA prediction records and the
      artifact contract (``manifest.json`` + ``predictions.jsonl``).
    - :mod:`aic2026.qa.qa_dataset`: QA query selection and text resolution.
    - :mod:`aic2026.qa.qa_prediction_runner`: retrieval-driven QA prediction
      generation reusing the shared index retriever and candidate
      foundation.
    - :mod:`aic2026.qa.qa_submission`: isolated provisional BTC submission
      export.
    - :mod:`aic2026.qa.qa_scoring`: placeholder for the future QA scorer.
"""

from aic2026.qa.qa_prediction_runner import (
    QaPredictionRunnerConfig,
    load_qa_predictions_by_query,
    read_qa_artifact,
    run_qa_prediction,
    write_qa_artifact,
)
from aic2026.qa.qa_scoring import QA_SCORING_STATUS, evaluate_qa
from aic2026.qa.qa_submission import (
    QA_SUBMISSION_CSV_COLUMNS,
    QA_SUBMISSION_FORMAT,
    build_submission_rows,
    write_qa_submission_csv,
)
from aic2026.qa.qa_types import (
    QA_PREDICTION_SCHEMA_VERSION,
    QaPrediction,
    QaPredictionArtifact,
    QaPredictionManifest,
    QaQueryResult,
    QueryTextField,
)

__all__ = [
    "QA_PREDICTION_SCHEMA_VERSION",
    "QA_SCORING_STATUS",
    "QA_SUBMISSION_CSV_COLUMNS",
    "QA_SUBMISSION_FORMAT",
    "QaPrediction",
    "QaPredictionArtifact",
    "QaPredictionManifest",
    "QaPredictionRunnerConfig",
    "QaQueryResult",
    "QueryTextField",
    "build_submission_rows",
    "evaluate_qa",
    "load_qa_predictions_by_query",
    "read_qa_artifact",
    "run_qa_prediction",
    "write_qa_artifact",
    "write_qa_submission_csv",
]