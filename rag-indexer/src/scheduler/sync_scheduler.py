"""APScheduler 기반 동기화 작업 스케줄러.

등록된 모든 소스에 대한 자동 동기화 작업을 관리합니다.

주요 기능:
    - Cron 표현식 기반 주기적 동기화
    - 수동 즉시 동기화 트리거
    - 멀티 워커를 통한 병렬 처리
    - 소스 타입별 커넥터 자동 선택
    - 작업 상태 추적 및 에러 관리
"""

from datetime import UTC, datetime
from typing import Callable, Optional

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..config import get_settings
from ..connectors import NotionConnector, SwaggerConnector
from ..logging_config import Loggers
from ..models import Source, SourceType, SyncJob, SyncJobStatus, SyncJobTrigger
from ..services import get_indexer
from ..storage import get_storage

logger = Loggers.scheduler()


class SyncScheduler:
    """자동 소스 동기화 스케줄러.

    APScheduler를 사용하여 Cron 트리거 기반의
    주기적 동기화를 수행합니다.

    주요 기능:
        - 백그라운드 스케줄러로 비차단 실행
        - ThreadPoolExecutor로 병렬 작업 처리
        - 누락된 실행을 하나로 병합 (coalesce)
        - 작업 인스턴스 중복 방지

    Attributes:
        cron_expression: 동기화 스케줄 Cron 표현식.
        max_workers: 최대 동시 동기화 작업 수.
    """

    def __init__(
        self,
        cron_expression: str = "0 * * * *",  # 기본값: 매시 정각
        max_workers: int = 2,
    ):
        """스케줄러를 초기화합니다.

        Args:
            cron_expression: 동기화 스케줄 Cron 표현식.
                기본값은 "0 * * * *" (매시 정각).
            max_workers: 최대 동시 동기화 작업 수.
                기본값은 2개.
        """
        self.cron_expression = cron_expression
        self.max_workers = max_workers
        self._scheduler: Optional[BackgroundScheduler] = None
        self._running = False

    @property
    def scheduler(self) -> BackgroundScheduler:
        """스케줄러를 지연 초기화합니다.

        첫 접근 시에만 BackgroundScheduler를 생성합니다.
        메모리 기반 작업 저장소와 스레드풀 실행기를 사용합니다.

        설정:
            - coalesce: 누락된 여러 실행을 하나로 병합
            - max_instances: 작업당 하나의 인스턴스만 허용
            - misfire_grace_time: 1시간의 유예 기간

        Returns:
            설정된 BackgroundScheduler 인스턴스.
        """
        if self._scheduler is None:
            self._scheduler = BackgroundScheduler(
                jobstores={
                    "default": MemoryJobStore(),
                },
                executors={
                    "default": ThreadPoolExecutor(self.max_workers),
                },
                job_defaults={
                    "coalesce": True,  # 누락된 여러 실행을 하나로 병합
                    "max_instances": 1,  # 작업당 하나의 인스턴스만
                    "misfire_grace_time": 3600,  # 1시간 유예 기간
                },
            )
        return self._scheduler

    def start(self) -> None:
        """스케줄러를 시작합니다.

        메인 동기화 작업을 추가하고 스케줄러를 시작합니다.
        이미 실행 중인 경우 경고 로그를 출력합니다.

        동작:
            1. 이미 실행 중인지 확인
            2. Cron 트리거로 메인 동기화 작업 추가
            3. 스케줄러 시작
            4. 시작 로그 출력
        """
        if self._running:
            logger.warning("스케줄러가 이미 실행 중입니다")
            return

        # 메인 동기화 작업 추가
        trigger = CronTrigger.from_crontab(self.cron_expression)
        self.scheduler.add_job(
            func=self._run_sync_all,
            trigger=trigger,
            id="sync_all_sources",
            name="Sync All Sources",
            replace_existing=True,
        )

        self.scheduler.start()
        self._running = True
        logger.info(
            "스케줄러 시작됨",
            cron=self.cron_expression,
            max_workers=self.max_workers,
        )

    def stop(self) -> None:
        """스케줄러를 중지합니다.

        실행 중인 작업이 완료될 때까지 기다린 후 종료합니다.
        """
        if not self._running:
            return

        self.scheduler.shutdown(wait=True)
        self._running = False
        logger.info("스케줄러 중지됨")

    def trigger_sync(
        self,
        source_id: Optional[str] = None,
        callback: Optional[Callable[[SyncJob], None]] = None,
    ) -> SyncJob:
        """즉시 동기화 작업을 트리거합니다.

        수동으로 동기화를 시작할 때 사용합니다.
        스케줄된 작업과 달리 동기적으로 실행됩니다.

        Args:
            source_id: 동기화할 소스 ID.
                None이면 모든 소스를 동기화합니다.
            callback: 작업 완료 시 호출할 콜백 함수.
                SyncJob 객체를 인자로 받습니다.

        Returns:
            생성된 SyncJob 객체.
        """
        job = SyncJob(
            source_id=source_id,
            trigger=SyncJobTrigger.MANUAL,
        )

        # 수동 트리거는 동기적으로 실행
        self._execute_sync(job, callback)
        return job

    def get_next_run(self) -> Optional[datetime]:
        """다음 예정된 실행 시간을 반환합니다.

        스케줄러가 실행 중이 아니거나 작업이 없으면
        None을 반환합니다.

        Returns:
            다음 실행 datetime 또는 None.
        """
        if not self._running:
            return None

        job = self.scheduler.get_job("sync_all_sources")
        if job and job.next_run_time:
            return job.next_run_time
        return None

    def is_running(self) -> bool:
        """스케줄러가 실행 중인지 확인합니다.

        Returns:
            실행 중이면 True.
        """
        return self._running

    # ==================== 내부 메서드 ====================

    def _run_sync_all(self) -> None:
        """모든 소스에 대해 동기화를 실행합니다 (스케줄된 작업).

        Cron 트리거에 의해 자동으로 호출됩니다.
        SyncJobTrigger.SCHEDULED로 작업을 생성합니다.
        """
        job = SyncJob(
            source_id=None,  # 모든 소스
            trigger=SyncJobTrigger.SCHEDULED,
        )
        self._execute_sync(job)

    def _execute_sync(
        self,
        job: SyncJob,
        callback: Optional[Callable[[SyncJob], None]] = None,
    ) -> None:
        """동기화 작업을 실행합니다.

        전체 동기화 파이프라인을 오케스트레이션합니다:
        1. 작업 시작 및 저장소에 기록
        2. 대상 소스 조회 (전체 또는 특정)
        3. 각 소스별 동기화 실행
        4. 작업 완료/실패 상태 업데이트
        5. 콜백 호출

        Args:
            job: 실행할 SyncJob 객체.
            callback: 완료 시 호출할 콜백 함수.
        """
        storage = get_storage()
        indexer = get_indexer()
        settings = get_settings()

        job.start()
        storage.add_sync_job(job)

        logger.info(
            "동기화 작업 시작",
            job_id=job.id,
            source_id=job.source_id,
            trigger=job.trigger.value,
        )

        try:
            # 동기화할 소스 조회
            if job.source_id:
                source = storage.get_source(job.source_id)
                if source:
                    sources = [source]
                else:
                    job.fail(f"소스를 찾을 수 없음: {job.source_id}")
                    storage.update_sync_job(job)
                    return
            else:
                sources = storage.get_sources()

            if not sources:
                logger.info("동기화할 소스가 없습니다")
                job.complete()
                storage.update_sync_job(job)
                return

            # 각 소스 처리
            has_errors = False
            for source in sources:
                try:
                    self._sync_source(source, job, indexer, storage, settings)
                except Exception as e:
                    has_errors = True
                    job.add_error(
                        error_type=type(e).__name__,
                        message=str(e),
                        source_id=source.id,
                        retryable=True,
                    )
                    logger.error(
                        "소스 동기화 실패",
                        source_id=source.id,
                        error=str(e),
                    )

            # 작업 완료로 표시
            job.complete(partial=has_errors)
            storage.update_sync_job(job)

            logger.info(
                "동기화 작업 완료",
                job_id=job.id,
                status=job.status.value,
                documents_processed=job.documents_processed,
                chunks_created=job.chunks_created,
                errors=len(job.errors),
            )

        except Exception as e:
            job.fail(str(e))
            storage.update_sync_job(job)
            logger.error("동기화 작업 실패", job_id=job.id, error=str(e))

        finally:
            if callback:
                callback(job)

    def _sync_source(
        self,
        source: Source,
        job: SyncJob,
        indexer,
        storage,
        settings,
    ) -> None:
        """단일 소스를 동기화합니다.

        소스 타입에 따라 적절한 커넥터를 선택하고,
        새 문서, 업데이트된 문서, 삭제된 문서를 처리합니다.

        처리 순서:
            1. 기존 문서 조회
            2. 소스 타입별 커넥터로 변경사항 조회
            3. 새 문서 인덱싱
            4. 업데이트된 문서 재인덱싱
            5. 삭제된 문서 청크 제거
            6. 소스 last_synced 타임스탬프 업데이트

        Args:
            source: 동기화할 소스 객체.
            job: 부모 SyncJob 객체.
            indexer: Indexer 인스턴스.
            storage: Storage 인스턴스.
            settings: 애플리케이션 설정.

        Raises:
            ValueError: 지원하지 않는 소스 타입인 경우.
        """
        logger.info("소스 동기화 중", source_id=source.id, type=source.source_type.value)

        # 기존 문서 조회
        existing_docs = storage.get_documents(source.id)

        # 소스 타입에 따라 새 문서 가져오기
        if source.source_type == SourceType.NOTION:
            connector = NotionConnector(api_key=settings.notion.api_key)
            new_docs, updated_docs, deleted_ids = connector.fetch_documents(
                source=source,
                existing_docs=existing_docs,
            )
        elif source.source_type == SourceType.SWAGGER:
            connector = SwaggerConnector()
            new_docs, updated_docs, deleted_ids = connector.fetch_documents(
                source=source,
                existing_docs=existing_docs,
            )
        else:
            raise ValueError(f"지원하지 않는 소스 타입: {source.source_type}")

        # 새 문서 처리
        for doc in new_docs:
            try:
                chunks = indexer.index_document(doc, source)
                job.documents_created += 1
                job.chunks_created += chunks
            except Exception as e:
                job.add_error(
                    error_type=type(e).__name__,
                    message=str(e),
                    document_id=doc.id,
                    source_id=source.id,
                )

        # 업데이트된 문서 처리
        for doc in updated_docs:
            try:
                chunks = indexer.index_document(doc, source)
                job.documents_updated += 1
                job.chunks_created += chunks
            except Exception as e:
                job.add_error(
                    error_type=type(e).__name__,
                    message=str(e),
                    document_id=doc.id,
                    source_id=source.id,
                )

        # 삭제된 문서 처리
        for doc_id in deleted_ids:
            try:
                indexer.delete_document_chunks(doc_id)
                storage.delete_document(doc_id)
                job.documents_deleted += 1
            except Exception as e:
                job.add_error(
                    error_type=type(e).__name__,
                    message=str(e),
                    document_id=doc_id,
                    source_id=source.id,
                )

        job.documents_processed += len(new_docs) + len(updated_docs) + len(deleted_ids)

        # 소스 last_synced 타임스탬프 업데이트
        source.last_synced = datetime.now(UTC)
        storage.update_source(source)


# 모듈 레벨 싱글톤 인스턴스
_scheduler: Optional[SyncScheduler] = None


def get_scheduler(
    cron_expression: Optional[str] = None,
) -> SyncScheduler:
    """스케줄러 인스턴스를 반환합니다.

    싱글톤 패턴으로 구현되어 있어
    애플리케이션 전체에서 하나의 인스턴스만 사용합니다.

    Args:
        cron_expression: Cron 표현식 오버라이드.
            None이면 설정에서 가져옵니다.

    Returns:
        SyncScheduler 인스턴스.

    Note:
        첫 호출 시 설정에서 cron_expression과
        max_workers를 읽어 인스턴스를 생성합니다.
    """
    global _scheduler

    if _scheduler is None:
        settings = get_settings()
        cron = cron_expression or settings.scheduler.cron_expression
        _scheduler = SyncScheduler(
            cron_expression=cron,
            max_workers=settings.scheduler.max_workers,
        )

    return _scheduler
