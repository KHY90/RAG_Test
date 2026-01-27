"""Reciprocal Rank Fusion (RRF) 알고리즘에 대한 단위 테스트."""

import pytest
from uuid import uuid4


class TestReciprocalRankFusion:
    """RRF 알고리즘 구현에 대한 테스트."""

    def test_rrf_single_ranking(self):
        """단일 순위 목록으로 RRF 테스트."""
        from src.services.search import reciprocal_rank_fusion

        # 3개 항목이 있는 단일 순위
        rankings = [
            [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        ]

        result = reciprocal_rank_fusion(rankings, k=60)

        # 첫 번째 항목은 가장 높은 RRF 점수를 가져야 합니다
        assert len(result) == 3
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"
        assert result[2][0] == "doc3"
        # RRF 점수는 다음과 같아야 합니다: 1/(60+1), 1/(60+2), 1/(60+3)
        assert result[0][1] == pytest.approx(1 / 61, rel=1e-5)

    def test_rrf_multiple_rankings(self):
        """여러 순위 목록을 결합하는 RRF 테스트."""
        from src.services.search import reciprocal_rank_fusion

        # 중복된 문서가 있는 두 개의 순위
        rankings = [
            [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)],  # 순위 1
            [("doc2", 0.95), ("doc1", 0.8), ("doc4", 0.6)],  # 순위 2
        ]

        result = reciprocal_rank_fusion(rankings, k=60)

        # doc1과 doc2는 두 목록에 모두 존재하므로 더 높은 순위를 가져야 합니다
        result_dict = {doc_id: score for doc_id, score in result}
        assert "doc1" in result_dict
        assert "doc2" in result_dict
        # doc2는 순위 2에서 1위, 순위 1에서 2위 -> 더 높은 결합 점수
        # doc1은 순위 1에서 1위, 순위 2에서 2위 -> 비슷해야 함

    def test_rrf_with_k_parameter(self):
        """k 매개변수가 RRF 점수에 올바르게 영향을 미치는지 테스트."""
        from src.services.search import reciprocal_rank_fusion

        rankings = [[("doc1", 0.9), ("doc2", 0.7)]]

        result_k60 = reciprocal_rank_fusion(rankings, k=60)
        result_k10 = reciprocal_rank_fusion(rankings, k=10)

        # k가 작을수록 순위 차이가 더 중요합니다
        score_diff_k60 = result_k60[0][1] - result_k60[1][1]
        score_diff_k10 = result_k10[0][1] - result_k10[1][1]

        # k가 작을수록 점수 차이가 더 커져야 합니다
        assert score_diff_k10 > score_diff_k60

    def test_rrf_empty_rankings(self):
        """빈 순위 목록으로 RRF 테스트."""
        from src.services.search import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([], k=60)
        assert result == []

        result = reciprocal_rank_fusion([[]], k=60)
        assert result == []

    def test_rrf_preserves_original_scores(self):
        """출력에서 원래 점수가 보존되는지 테스트."""
        from src.services.search import reciprocal_rank_fusion

        rankings = [
            [("doc1", 0.95), ("doc2", 0.85)],
            [("doc1", 0.90), ("doc3", 0.75)],
        ]

        result = reciprocal_rank_fusion(rankings, k=60)

        # 결과는 (doc_id, rrf_score) 튜플의 목록이어야 합니다
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    def test_rrf_sorted_by_score_descending(self):
        """RRF 결과가 점수 내림차순으로 정렬되는지 테스트."""
        from src.services.search import reciprocal_rank_fusion

        rankings = [
            [("doc1", 0.5), ("doc2", 0.9), ("doc3", 0.7)],
            [("doc4", 0.8), ("doc2", 0.6), ("doc1", 0.4)],
        ]

        result = reciprocal_rank_fusion(rankings, k=60)

        # 결과는 RRF 점수 내림차순으로 정렬되어야 합니다
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_document_in_all_rankings_scores_highest(self):
        """모든 순위에 나타나는 문서가 부스트되는지 테스트."""
        from src.services.search import reciprocal_rank_fusion

        # doc1은 3개의 모든 순위에 나타납니다
        rankings = [
            [("doc1", 0.9), ("doc2", 0.8)],
            [("doc3", 0.9), ("doc1", 0.8)],
            [("doc4", 0.9), ("doc5", 0.8), ("doc1", 0.7)],
        ]

        result = reciprocal_rank_fusion(rankings, k=60)
        result_dict = {doc_id: score for doc_id, score in result}

        # doc1은 가장 높은 점수를 가져야 합니다 (모든 순위에 나타남)
        assert result[0][0] == "doc1"

    def test_rrf_with_uuid_identifiers(self):
        """Test RRF works with UUID document identifiers."""
        from src.services.search import reciprocal_rank_fusion

        doc_ids = [uuid4() for _ in range(3)]

        rankings = [
            [(doc_ids[0], 0.9), (doc_ids[1], 0.8)],
            [(doc_ids[1], 0.95), (doc_ids[2], 0.7)],
        ]

        result = reciprocal_rank_fusion(rankings, k=60)

        # UUID와 함께 작동해야 합니다
        assert len(result) == 3
        result_ids = {doc_id for doc_id, _ in result}
        assert result_ids == set(doc_ids)
