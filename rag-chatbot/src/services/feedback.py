"""피드백 서비스 모듈.

Redis를 사용하여 사용자 피드백 데이터를 저장합니다.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import redis

from ..config import get_settings
from ..models import Feedback

logger = logging.getLogger(__name__)


class FeedbackService:
    """피드백 서비스 클래스.

    Redis에 답변 품질 피드백을 저장하고 조회합니다.
    선택적으로 JSON 파일로 백업할 수 있습니다.
    """

    # Redis 키 패턴
    FEEDBACK_KEY_PREFIX = "feedback:"
    FEEDBACK_INDEX_KEY = "feedback:index"

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
    ) -> None:
        """FeedbackService 초기화.

        Args:
            redis_client: Redis 클라이언트 (None이면 자동 생성)
        """
        self.settings = get_settings()

        if redis_client:
            self._redis = redis_client
        else:
            self._redis = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                decode_responses=True,
            )

        logger.info("FeedbackService 초기화 완료")

    def save(self, feedback: Feedback) -> bool:
        """피드백 저장.

        Args:
            feedback: 저장할 피드백

        Returns:
            저장 성공 여부
        """
        key = Feedback.redis_key(feedback.message_ts)

        try:
            data = feedback.model_dump_json()
            # 피드백 저장 (TTL 없음 - 영구 보관)
            self._redis.set(key, data)

            # 인덱스에 추가 (시간순 정렬용)
            self._redis.zadd(
                self.FEEDBACK_INDEX_KEY,
                {feedback.message_ts: feedback.created_at.timestamp()},
            )

            logger.info(
                f"피드백 저장 완료 (message_ts={feedback.message_ts}, "
                f"rating={feedback.rating}, reaction={feedback.reaction})"
            )
            return True
        except Exception as e:
            logger.error(f"피드백 저장 실패 (message_ts={feedback.message_ts}): {e}")
            return False

    def get(self, message_ts: str) -> Feedback | None:
        """피드백 조회.

        Args:
            message_ts: 메시지 타임스탬프

        Returns:
            피드백 (없으면 None)
        """
        key = Feedback.redis_key(message_ts)

        try:
            data = self._redis.get(key)
            if data:
                return Feedback.model_validate_json(data)
            return None
        except Exception as e:
            logger.error(f"피드백 조회 실패 (message_ts={message_ts}): {e}")
            return None

    def delete(self, message_ts: str) -> bool:
        """피드백 삭제.

        Args:
            message_ts: 메시지 타임스탬프

        Returns:
            삭제 성공 여부
        """
        key = Feedback.redis_key(message_ts)

        try:
            self._redis.delete(key)
            self._redis.zrem(self.FEEDBACK_INDEX_KEY, message_ts)
            logger.debug(f"피드백 삭제 완료 (message_ts={message_ts})")
            return True
        except Exception as e:
            logger.error(f"피드백 삭제 실패 (message_ts={message_ts}): {e}")
            return False

    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Feedback]:
        """모든 피드백 조회.

        시간 역순으로 정렬됩니다.

        Args:
            limit: 최대 조회 수
            offset: 시작 위치

        Returns:
            피드백 목록
        """
        try:
            # 인덱스에서 message_ts 목록 조회 (최신순)
            message_ts_list = self._redis.zrevrange(
                self.FEEDBACK_INDEX_KEY,
                offset,
                offset + limit - 1,
            )

            feedbacks = []
            for message_ts in message_ts_list:
                feedback = self.get(message_ts)
                if feedback:
                    feedbacks.append(feedback)

            return feedbacks
        except Exception as e:
            logger.error(f"전체 피드백 조회 실패: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """피드백 통계 조회.

        Returns:
            통계 정보 딕셔너리
        """
        try:
            all_feedbacks = self.get_all(limit=1000)

            total = len(all_feedbacks)
            positive = sum(1 for f in all_feedbacks if f.rating == "positive")
            negative = sum(1 for f in all_feedbacks if f.rating == "negative")

            return {
                "total": total,
                "positive": positive,
                "negative": negative,
                "positive_ratio": positive / total if total > 0 else 0.0,
            }
        except Exception as e:
            logger.error(f"피드백 통계 조회 실패: {e}")
            return {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "positive_ratio": 0.0,
                "error": str(e),
            }

    def export_to_json(
        self,
        filepath: str | Path | None = None,
    ) -> str:
        """피드백 데이터 JSON 파일로 내보내기.

        Args:
            filepath: 저장할 파일 경로 (기본: feedback_export_{timestamp}.json)

        Returns:
            저장된 파일 경로
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"feedback_export_{timestamp}.json")
        else:
            filepath = Path(filepath)

        try:
            all_feedbacks = self.get_all(limit=10000)

            export_data = {
                "exported_at": datetime.now().isoformat(),
                "total_count": len(all_feedbacks),
                "feedbacks": [f.to_export_dict() for f in all_feedbacks],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"피드백 내보내기 완료 (filepath={filepath}, count={len(all_feedbacks)})"
            )
            return str(filepath)
        except Exception as e:
            logger.error(f"피드백 내보내기 실패: {e}")
            raise

    def health_check(self) -> dict:
        """서비스 상태 확인.

        Returns:
            상태 정보 딕셔너리
        """
        try:
            self._redis.ping()
            total_count = self._redis.zcard(self.FEEDBACK_INDEX_KEY)
            return {
                "status": "healthy",
                "redis": {
                    "connected": True,
                    "host": self.settings.redis_host,
                    "port": self.settings.redis_port,
                },
                "feedback_count": total_count,
            }
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# 기본 인스턴스 (지연 로딩)
_default_service: FeedbackService | None = None


def get_feedback_service() -> FeedbackService:
    """FeedbackService 싱글톤 인스턴스 반환.

    Returns:
        FeedbackService 인스턴스
    """
    global _default_service
    if _default_service is None:
        _default_service = FeedbackService()
    return _default_service
