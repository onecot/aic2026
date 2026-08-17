"""Tests for the QA scoring placeholder contract."""

from __future__ import annotations

import pytest

from aic2026.evaluation import EvaluationSet
from aic2026.qa.qa_scoring import QA_SCORING_STATUS, evaluate_qa
from aic2026.qa.qa_types import QaPrediction


def test_scoring_status_is_not_implemented() -> None:
    assert QA_SCORING_STATUS == "not-implemented"


def test_evaluate_qa_always_raises_not_implemented() -> None:
    evaluation_set = EvaluationSet(queries=())
    predictions = {
        "qa_001": [QaPrediction(rank=1, video_id="L21_V001", frame_index=10, score=0.9)]
    }
    with pytest.raises(NotImplementedError, match="intentionally not implemented"):
        evaluate_qa(evaluation_set, predictions)


def test_evaluate_qa_validates_contract_before_raising() -> None:
    with pytest.raises(TypeError, match="evaluation_set must be an EvaluationSet"):
        evaluate_qa("not-a-set", {})
    with pytest.raises(TypeError, match="predictions_by_query must be a Mapping"):
        evaluate_qa(EvaluationSet(queries=()), [])
    with pytest.raises(TypeError, match="values must be a list or tuple"):
        evaluate_qa(EvaluationSet(queries=()), {"qa_001": "not-a-list"})
    with pytest.raises(TypeError, match="contain only QaPrediction"):
        evaluate_qa(EvaluationSet(queries=()), {"qa_001": ["not-a-prediction"]})
    with pytest.raises(ValueError, match="non-empty query ids"):
        evaluate_qa(EvaluationSet(queries=()), {"": []})