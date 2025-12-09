"""RAG Indexer 스케줄러 모듈.

자동 동기화 작업을 관리하는 스케줄러 클래스를 제공합니다.

주요 기능:
    - APScheduler 기반 백그라운드 작업 스케줄링
    - Cron 표현식을 통한 주기적 동기화
    - 수동 동기화 트리거 지원
    - 소스별 또는 전체 소스 동기화

Exports:
    SyncScheduler: 소스 동기화 스케줄러 클래스.
        Cron 트리거를 사용한 주기적 동기화를 관리합니다.
    get_scheduler: 스케줄러 싱글톤 인스턴스를 반환합니다.
        설정에서 cron 표현식과 워커 수를 가져옵니다.
"""

from .sync_scheduler import SyncScheduler, get_scheduler

__all__ = [
    "SyncScheduler",
    "get_scheduler",
]
