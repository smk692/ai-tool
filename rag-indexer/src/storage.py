"""JSON 파일 기반 상태 저장소.

JSON 파일을 사용한 영구 상태 관리:
    - sources.json: 등록된 데이터 소스
    - documents.json: 문서 메타데이터 및 해시
    - sync_history.json: 동기화 작업 이력 (최근 100개)

주요 기능:
    - 소스 CRUD 작업 (등록, 조회, 수정, 삭제)
    - 문서 메타데이터 관리
    - 동기화 이력 추적
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from .models import Document, Source, SourceType, SyncJob


class Storage:
    """인덱서 상태를 위한 JSON 파일 기반 저장소.

    소스, 문서, 동기화 이력을 JSON 파일에 저장합니다.
    파일 수준의 작업을 통해 스레드 안전성을 보장합니다.

    Attributes:
        MAX_SYNC_HISTORY: 유지할 최대 동기화 이력 수 (100개).
        SOURCES_FILE: 소스 정보 파일명.
        DOCUMENTS_FILE: 문서 정보 파일명.
        SYNC_HISTORY_FILE: 동기화 이력 파일명.
    """

    MAX_SYNC_HISTORY = 100
    SOURCES_FILE = "sources.json"
    DOCUMENTS_FILE = "documents.json"
    SYNC_HISTORY_FILE = "sync_history.json"

    def __init__(self, data_dir: Path | str):
        """저장소를 초기화합니다.

        Args:
            data_dir: JSON 상태 파일을 저장할 디렉토리.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._sources_path = self.data_dir / self.SOURCES_FILE
        self._documents_path = self.data_dir / self.DOCUMENTS_FILE
        self._sync_history_path = self.data_dir / self.SYNC_HISTORY_FILE

    # ==================== 소스 관리 ====================

    def get_sources(self) -> list[Source]:
        """등록된 모든 소스를 조회합니다.

        Returns:
            Source 객체 목록.
        """
        data = self._load_json(self._sources_path, {"sources": []})
        return [Source.from_json_safe(s) for s in data.get("sources", [])]

    def get_source(self, source_id: str) -> Source | None:
        """ID로 소스를 조회합니다.

        Args:
            source_id: 찾을 소스 ID.

        Returns:
            찾은 경우 Source 객체, 없으면 None.
        """
        sources = self.get_sources()
        for source in sources:
            if source.id == source_id:
                return source
        return None

    def get_source_by_name(self, name: str) -> Source | None:
        """이름으로 소스를 조회합니다.

        Args:
            name: 찾을 소스 이름.

        Returns:
            찾은 경우 Source 객체, 없으면 None.
        """
        sources = self.get_sources()
        for source in sources:
            if source.name == name:
                return source
        return None

    def get_sources_by_type(self, source_type: SourceType) -> list[Source]:
        """타입으로 소스를 필터링하여 조회합니다.

        Args:
            source_type: 필터링할 소스 타입.

        Returns:
            일치하는 Source 목록.
        """
        return [s for s in self.get_sources() if s.source_type == source_type]

    def add_source(self, source: Source) -> Source:
        """새 소스를 추가합니다.

        Args:
            source: 추가할 소스.

        Returns:
            추가된 소스.

        Raises:
            ValueError: 동일한 이름의 소스가 존재하는 경우.
        """
        sources = self.get_sources()

        # 중복 이름 확인
        if any(s.name == source.name for s in sources):
            raise ValueError(f"Source with name '{source.name}' already exists")

        sources.append(source)
        self._save_sources(sources)
        return source

    def update_source(self, source: Source) -> Source:
        """기존 소스를 업데이트합니다.

        Args:
            source: 업데이트된 데이터가 포함된 소스.

        Returns:
            업데이트된 소스.

        Raises:
            ValueError: 소스를 찾을 수 없는 경우.
        """
        sources = self.get_sources()

        for i, s in enumerate(sources):
            if s.id == source.id:
                source.updated_at = datetime.now(UTC)
                sources[i] = source
                self._save_sources(sources)
                return source

        raise ValueError(f"Source with ID '{source.id}' not found")

    def delete_source(self, source_id: str) -> bool:
        """소스와 관련 문서를 삭제합니다.

        Args:
            source_id: 삭제할 소스 ID.

        Returns:
            삭제 성공 시 True, 없으면 False.
        """
        sources = self.get_sources()
        original_count = len(sources)

        sources = [s for s in sources if s.id != source_id]

        if len(sources) < original_count:
            self._save_sources(sources)
            # 관련 문서도 삭제
            self.delete_documents_by_source(source_id)
            return True

        return False

    def _save_sources(self, sources: list[Source]) -> None:
        """소스 목록을 파일에 저장합니다."""
        data = {"sources": [s.model_dump_json_safe() for s in sources]}
        self._save_json(self._sources_path, data)

    # ==================== 문서 관리 ====================

    def get_documents(self, source_id: str | None = None) -> list[Document]:
        """문서를 조회합니다. 소스별 필터링 가능.

        Args:
            source_id: 필터링할 소스 ID (선택적).

        Returns:
            Document 객체 목록.
        """
        data = self._load_json(self._documents_path, {"documents": []})
        documents = [Document.from_json_safe(d) for d in data.get("documents", [])]

        if source_id:
            documents = [d for d in documents if d.source_id == source_id]

        return documents

    def get_document(self, document_id: str) -> Document | None:
        """ID로 문서를 조회합니다.

        Args:
            document_id: 찾을 문서 ID.

        Returns:
            찾은 경우 Document 객체, 없으면 None.
        """
        documents = self.get_documents()
        for doc in documents:
            if doc.id == document_id:
                return doc
        return None

    def get_document_by_external_id(
        self, source_id: str, external_id: str
    ) -> Document | None:
        """외부 ID로 특정 소스 내 문서를 조회합니다.

        Args:
            source_id: 소스 ID.
            external_id: 외부 시스템 ID.

        Returns:
            찾은 경우 Document 객체, 없으면 None.
        """
        documents = self.get_documents(source_id)
        for doc in documents:
            if doc.external_id == external_id:
                return doc
        return None

    def upsert_document(self, document: Document) -> Document:
        """문서를 삽입하거나 업데이트합니다.

        Args:
            document: 삽입/업데이트할 문서.

        Returns:
            처리된 문서.
        """
        documents = self.get_documents()

        for i, doc in enumerate(documents):
            if doc.id == document.id:
                document.updated_at = datetime.now(UTC)
                documents[i] = document
                self._save_documents(documents)
                return document

        # 새 문서
        documents.append(document)
        self._save_documents(documents)
        return document

    def delete_document(self, document_id: str) -> bool:
        """문서를 삭제합니다.

        Args:
            document_id: 삭제할 문서 ID.

        Returns:
            삭제 성공 시 True, 없으면 False.
        """
        documents = self.get_documents()
        original_count = len(documents)

        documents = [d for d in documents if d.id != document_id]

        if len(documents) < original_count:
            self._save_documents(documents)
            return True

        return False

    def delete_documents_by_source(self, source_id: str) -> int:
        """특정 소스의 모든 문서를 삭제합니다.

        Args:
            source_id: 소스 ID.

        Returns:
            삭제된 문서 수.
        """
        documents = self.get_documents()
        original_count = len(documents)

        documents = [d for d in documents if d.source_id != source_id]
        deleted = original_count - len(documents)

        if deleted > 0:
            self._save_documents(documents)

        return deleted

    def _save_documents(self, documents: list[Document]) -> None:
        """문서 목록을 파일에 저장합니다."""
        data = {"documents": [d.model_dump_json_safe() for d in documents]}
        self._save_json(self._documents_path, data)

    # ==================== 동기화 이력 ====================

    def get_sync_history(self, limit: int = 100) -> list[SyncJob]:
        """최근 동기화 작업 이력을 조회합니다.

        Args:
            limit: 반환할 최대 작업 수.

        Returns:
            SyncJob 객체 목록 (최신순).
        """
        data = self._load_json(self._sync_history_path, {"jobs": []})
        jobs = [SyncJob.from_json_safe(j) for j in data.get("jobs", [])]

        # started_at 기준 내림차순 정렬 (최신순)
        jobs.sort(key=lambda j: j.started_at or datetime.min, reverse=True)

        return jobs[:limit]

    def add_sync_job(self, job: SyncJob) -> SyncJob:
        """동기화 작업을 이력에 추가합니다.

        Args:
            job: 추가할 SyncJob.

        Returns:
            추가된 작업.
        """
        jobs = self.get_sync_history(self.MAX_SYNC_HISTORY - 1)
        jobs.insert(0, job)

        # MAX_SYNC_HISTORY 개수만큼만 유지
        jobs = jobs[: self.MAX_SYNC_HISTORY]

        self._save_sync_history(jobs)
        return job

    def update_sync_job(self, job: SyncJob) -> SyncJob:
        """기존 동기화 작업을 업데이트합니다.

        Args:
            job: 업데이트된 데이터가 포함된 SyncJob.

        Returns:
            업데이트된 작업.
        """
        data = self._load_json(self._sync_history_path, {"jobs": []})
        jobs = [SyncJob.from_json_safe(j) for j in data.get("jobs", [])]

        for i, j in enumerate(jobs):
            if j.id == job.id:
                jobs[i] = job
                self._save_sync_history(jobs)
                return job

        # 작업을 찾을 수 없으면 추가
        return self.add_sync_job(job)

    def get_last_sync_job(self, source_id: str | None = None) -> SyncJob | None:
        """가장 최근 동기화 작업을 조회합니다.

        Args:
            source_id: 필터링할 소스 ID (선택적).

        Returns:
            가장 최근 SyncJob, 없으면 None.
        """
        jobs = self.get_sync_history()

        if source_id:
            jobs = [j for j in jobs if j.source_id == source_id]

        return jobs[0] if jobs else None

    def _save_sync_history(self, jobs: list[SyncJob]) -> None:
        """동기화 이력을 파일에 저장합니다."""
        data = {"jobs": [j.model_dump_json_safe() for j in jobs]}
        self._save_json(self._sync_history_path, data)

    # ==================== 유틸리티 ====================

    def _load_json(self, path: Path, default: dict) -> dict:
        """JSON 파일을 로드하거나 기본값을 반환합니다.

        Args:
            path: 파일 경로.
            default: 파일이 없을 때 반환할 기본값.

        Returns:
            로드된 데이터 또는 기본값.
        """
        if not path.exists():
            return default

        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return default

    def _save_json(self, path: Path, data: dict) -> None:
        """데이터를 JSON 파일에 저장합니다.

        Args:
            path: 파일 경로.
            data: 저장할 데이터.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# 전역 스토리지 인스턴스 (지연 초기화)
_storage: Storage | None = None


def get_storage(data_dir: Path | str | None = None) -> Storage:
    """스토리지 인스턴스를 반환합니다.

    Args:
        data_dir: 데이터 디렉토리 (선택적). 미지정 시 'data' 사용.

    Returns:
        Storage 인스턴스.
    """
    global _storage

    if data_dir is not None:
        return Storage(data_dir)

    if _storage is None:
        _storage = Storage(Path("data"))

    return _storage
