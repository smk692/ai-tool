# Feature Specification: RAG Document Indexer

**Feature Branch**: `004-rag-indexer`
**Created**: 2025-12-05
**Status**: Draft
**Input**: RAG 문서 등록 파이프라인 - Notion 데이터와 Swagger.json을 벡터DB에 등록

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Notion 문서 동기화 (Priority: P1)

운영자가 Notion 워크스페이스의 페이지와 데이터베이스를 벡터DB에 동기화하여, 나중에 Slack 챗봇이 해당 정보를 검색할 수 있게 합니다.

**Why this priority**: 핵심 MVP 기능. Notion은 팀의 주요 지식 저장소이며, 이 데이터 없이는 RAG 시스템이 작동하지 않습니다.

**Independent Test**: Notion API 키와 페이지 ID만 있으면 독립적으로 테스트 가능. 동기화 후 벡터DB에서 직접 검색하여 결과 확인 가능.

**Acceptance Scenarios**:

1. **Given** Notion API 키와 대상 페이지 ID가 설정되어 있고, **When** 동기화 명령을 실행하면, **Then** 해당 페이지의 모든 텍스트 콘텐츠가 벡터DB에 저장됩니다.

2. **Given** Notion 데이터베이스 ID가 설정되어 있고, **When** 동기화 명령을 실행하면, **Then** 데이터베이스의 모든 항목이 개별 문서로 벡터DB에 저장됩니다.

3. **Given** 이미 동기화된 Notion 페이지가 있고, **When** 페이지 내용이 변경된 후 재동기화하면, **Then** 변경된 내용만 업데이트됩니다.

---

### User Story 2 - Swagger API 문서 등록 (Priority: P2)

운영자가 외부 URL에서 Swagger/OpenAPI 문서를 가져와 벡터DB에 등록하여, API 관련 질문에 답변할 수 있게 합니다.

**Why this priority**: API 문서는 개발팀에게 중요한 정보원. Notion 다음으로 자주 검색되는 정보.

**Independent Test**: Swagger URL만 있으면 독립적으로 테스트 가능. 등록 후 API 엔드포인트 검색으로 결과 확인.

**Acceptance Scenarios**:

1. **Given** 유효한 Swagger JSON URL이 제공되고, **When** 등록 명령을 실행하면, **Then** 모든 API 엔드포인트가 벡터DB에 저장됩니다.

2. **Given** OpenAPI 3.0 형식의 문서가 제공되고, **When** 등록 명령을 실행하면, **Then** 각 엔드포인트의 경로, 메소드, 파라미터, 응답 스키마가 검색 가능한 형태로 저장됩니다.

3. **Given** 이미 등록된 Swagger 문서가 있고, **When** 새 버전의 URL로 재등록하면, **Then** 기존 데이터가 새 버전으로 교체됩니다.

---

### User Story 3 - 스케줄러 자동 동기화 (Priority: P3)

운영자가 주기적인 자동 동기화를 설정하여, 수동 개입 없이 최신 문서가 벡터DB에 반영되도록 합니다.

**Why this priority**: 수동 동기화가 작동한 후에 자동화 추가. 운영 편의성 향상.

**Independent Test**: 스케줄러 설정 후 지정된 시간에 동기화가 실행되는지 로그로 확인.

**Acceptance Scenarios**:

1. **Given** 동기화 스케줄이 "매일 오전 6시"로 설정되어 있고, **When** 해당 시간이 되면, **Then** 모든 등록된 소스가 자동으로 동기화됩니다.

2. **Given** 스케줄러가 동기화를 실행 중이고, **When** 특정 소스에서 에러가 발생하면, **Then** 다른 소스는 계속 처리되고 에러 내용이 기록됩니다.

---

### User Story 4 - 수동 트리거 CLI (Priority: P4)

운영자가 CLI 명령어로 특정 소스만 선택적으로 동기화하거나 전체 동기화를 실행합니다.

**Why this priority**: 디버깅과 긴급 업데이트에 필요. 스케줄러보다 우선순위 낮음.

**Independent Test**: CLI 명령어 실행으로 독립 테스트 가능.

**Acceptance Scenarios**:

1. **Given** CLI가 설치되어 있고, **When** `rag-indexer sync --source notion` 명령을 실행하면, **Then** Notion 소스만 동기화됩니다.

2. **Given** CLI가 설치되어 있고, **When** `rag-indexer sync --all` 명령을 실행하면, **Then** 모든 소스가 동기화됩니다.

3. **Given** CLI가 설치되어 있고, **When** `rag-indexer status` 명령을 실행하면, **Then** 각 소스의 마지막 동기화 시간과 문서 수가 표시됩니다.

---

### Edge Cases

- **Notion API 제한**: API 요청 한도 초과 시 지수 백오프로 재시도
- **대용량 페이지**: 텍스트가 매우 긴 페이지는 적절한 크기로 청킹
- **Swagger 파싱 실패**: 유효하지 않은 JSON 또는 지원하지 않는 버전 처리
- **네트워크 오류**: URL 접근 불가 시 재시도 및 알림
- **중복 문서**: 같은 문서 재등록 시 기존 데이터 업데이트 (중복 생성 방지)
- **빈 콘텐츠**: 텍스트가 없는 페이지/항목은 건너뛰고 로그 기록

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 시스템은 Notion API를 통해 페이지 콘텐츠를 읽을 수 있어야 합니다.
- **FR-002**: 시스템은 Notion 데이터베이스의 모든 항목을 개별 문서로 처리할 수 있어야 합니다.
- **FR-003**: 시스템은 URL에서 Swagger/OpenAPI JSON을 다운로드할 수 있어야 합니다.
- **FR-004**: 시스템은 OpenAPI 2.0 (Swagger) 및 3.0 형식을 파싱할 수 있어야 합니다.
- **FR-005**: 시스템은 긴 텍스트를 검색에 적합한 크기의 청크로 분할해야 합니다.
- **FR-006**: 시스템은 각 청크에 대해 벡터 임베딩을 생성해야 합니다.
- **FR-007**: 시스템은 임베딩과 메타데이터를 벡터 데이터베이스에 저장해야 합니다.
- **FR-008**: 시스템은 각 문서의 해시를 저장하여 변경 감지를 할 수 있어야 합니다.
- **FR-009**: 시스템은 cron 형식의 스케줄 설정을 지원해야 합니다.
- **FR-010**: 시스템은 동기화 실패 시 지정된 횟수만큼 재시도해야 합니다.
- **FR-011**: 시스템은 CLI를 통해 수동 동기화를 지원해야 합니다.
- **FR-012**: 시스템은 동기화 작업의 진행 상황과 결과를 로그로 기록해야 합니다.

### Key Entities

- **Source**: 데이터 소스 정의 (타입: Notion/Swagger, 식별자, 설정)
- **Document**: 원본 문서 (소스 참조, 제목, 콘텐츠 해시, 마지막 수정 시간)
- **Chunk**: 분할된 텍스트 조각 (문서 참조, 순서, 텍스트, 임베딩 벡터)
- **SyncJob**: 동기화 작업 (시작 시간, 종료 시간, 상태, 처리된 문서 수, 에러 내용)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 운영자가 5분 이내에 새 소스를 등록하고 첫 동기화를 완료할 수 있습니다.
- **SC-002**: 100개 Notion 페이지 동기화가 5분 이내에 완료됩니다.
- **SC-003**: 동기화 작업의 성공률이 99% 이상입니다 (네트워크/API 장애 제외).
- **SC-004**: 증분 업데이트 시 변경되지 않은 문서는 재처리하지 않아 전체 동기화 대비 80% 이상 시간 단축됩니다.
- **SC-005**: 스케줄러가 설정된 시간의 1분 이내에 동기화를 시작합니다.
- **SC-006**: 동기화된 문서의 검색 관련성 점수가 사용자 기대와 90% 이상 일치합니다.
