"""동기화 작업(SyncJob) 모델 정의.

동기화 작업 실행 기록을 나타냅니다.
"""

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class SyncJobStatus(str, Enum):
    """동기화 작업 상태.

    Attributes:
        PENDING: 대기 중
        RUNNING: 실행 중
        COMPLETED: 완료
        FAILED: 실패
        PARTIAL: 부분 완료 (일부 소스 성공, 일부 실패)
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # 일부 소스 성공, 일부 실패


class SyncJobTrigger(str, Enum):
    """동기화 작업 트리거 방식.

    Attributes:
        MANUAL: 수동 실행
        SCHEDULED: 스케줄 실행
    """

    MANUAL = "manual"
    SCHEDULED = "scheduled"


class SyncError(BaseModel):
    """동기화 작업 오류 기록.

    동기화 중 발생한 개별 오류를 기록합니다.

    Attributes:
        document_id: 오류가 발생한 문서 ID (선택적)
        source_id: 오류가 발생한 소스 ID (선택적)
        error_type: 오류 유형/클래스 이름
        message: 오류 메시지
        timestamp: 오류 발생 시간
        retryable: 재시도 가능 여부
    """

    document_id: str | None = Field(
        default=None,
        description="오류가 발생한 문서 ID",
    )
    source_id: str | None = Field(
        default=None,
        description="오류가 발생한 소스 ID",
    )
    error_type: str = Field(
        ...,
        description="오류 유형/클래스 이름",
    )
    message: str = Field(
        ...,
        description="오류 메시지",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="오류 발생 시간",
    )
    retryable: bool = Field(
        default=False,
        description="재시도 가능 여부",
    )


class SyncJob(BaseModel):
    """동기화 작업 기록.

    동기화 작업의 실행 및 결과를 추적합니다.

    Attributes:
        id: 고유 작업 식별자 (UUID)
        source_id: 대상 소스 ID (None이면 모든 소스)
        trigger: 작업 트리거 방식
        status: 현재 작업 상태
        started_at: 작업 시작 시간
        completed_at: 작업 완료 시간
        documents_processed: 처리된 총 문서 수
        documents_created: 생성된 새 문서 수
        documents_updated: 업데이트된 문서 수
        documents_deleted: 삭제된 문서 수
        documents_skipped: 건너뛴 문서 수 (변경 없음)
        chunks_created: 벡터DB에 생성된 청크 수
        errors: 발생한 오류 목록
        error_message: 실패 시 주요 오류 메시지
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="고유 작업 식별자 (UUID)",
    )
    source_id: str | None = Field(
        default=None,
        description="대상 소스 ID (None이면 모든 소스)",
    )
    trigger: SyncJobTrigger = Field(
        default=SyncJobTrigger.MANUAL,
        description="작업 트리거 방식",
    )
    status: SyncJobStatus = Field(
        default=SyncJobStatus.PENDING,
        description="현재 작업 상태",
    )
    started_at: datetime | None = Field(
        default=None,
        description="작업 시작 시간",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="작업 완료 시간",
    )

    # 통계
    documents_processed: int = Field(
        default=0,
        description="처리된 총 문서 수",
    )
    documents_created: int = Field(
        default=0,
        description="생성된 새 문서 수",
    )
    documents_updated: int = Field(
        default=0,
        description="업데이트된 문서 수",
    )
    documents_deleted: int = Field(
        default=0,
        description="삭제된 문서 수",
    )
    documents_skipped: int = Field(
        default=0,
        description="건너뛴 문서 수 (변경 없음)",
    )
    chunks_created: int = Field(
        default=0,
        description="벡터DB에 생성된 청크 수",
    )

    # 오류
    errors: list[SyncError] = Field(
        default_factory=list,
        description="발생한 오류 목록",
    )
    error_message: str | None = Field(
        default=None,
        description="실패 시 주요 오류 메시지",
    )

    def start(self) -> None:
        """작업을 시작 상태로 표시.

        status를 RUNNING으로, started_at을 현재 시간으로 설정합니다.
        """
        self.status = SyncJobStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def complete(self, partial: bool = False) -> None:
        """작업을 완료 상태로 표시.

        Args:
            partial: 일부 소스가 실패한 경우 True.
        """
        self.status = SyncJobStatus.PARTIAL if partial else SyncJobStatus.COMPLETED
        self.completed_at = datetime.now(UTC)

    def fail(self, message: str) -> None:
        """작업을 실패 상태로 표시.

        Args:
            message: 주요 오류 메시지.
        """
        self.status = SyncJobStatus.FAILED
        self.error_message = message
        self.completed_at = datetime.now(UTC)

    def add_error(
        self,
        error_type: str,
        message: str,
        document_id: str | None = None,
        source_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        """작업에 오류 추가.

        Args:
            error_type: 오류 유형/클래스 이름.
            message: 오류 메시지.
            document_id: 관련 문서 ID.
            source_id: 관련 소스 ID.
            retryable: 재시도 가능 여부.
        """
        self.errors.append(
            SyncError(
                error_type=error_type,
                message=message,
                document_id=document_id,
                source_id=source_id,
                retryable=retryable,
            )
        )

    @property
    def duration_seconds(self) -> float | None:
        """작업 소요 시간(초) 반환.

        Returns:
            시작과 완료 시간이 있으면 소요 시간(초), 없으면 None.
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def has_errors(self) -> bool:
        """오류 발생 여부 확인.

        Returns:
            오류가 하나라도 있으면 True.
        """
        return len(self.errors) > 0

    def model_dump_json_safe(self) -> dict:
        """JSON 직렬화 가능한 딕셔너리로 변환.

        datetime 필드를 ISO 포맷 문자열로 변환합니다.

        Returns:
            JSON 직렬화 가능한 딕셔너리.
        """
        data = self.model_dump()
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        data["errors"] = [
            {
                **e.model_dump(),
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self.errors
        ]
        return data

    @classmethod
    def from_json_safe(cls, data: dict) -> "SyncJob":
        """JSON-safe 딕셔너리에서 SyncJob 생성.

        ISO 포맷 문자열을 datetime 객체로 파싱합니다.

        Args:
            data: JSON-safe 딕셔너리 데이터.

        Returns:
            SyncJob 인스턴스.
        """
        if data.get("started_at") and isinstance(data["started_at"], str):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at") and isinstance(data["completed_at"], str):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])

        # 오류 객체 파싱
        errors = []
        for e in data.get("errors", []):
            if isinstance(e.get("timestamp"), str):
                e["timestamp"] = datetime.fromisoformat(e["timestamp"])
            errors.append(SyncError(**e))
        data["errors"] = errors

        return cls(**data)
