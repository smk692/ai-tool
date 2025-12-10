# Implementation Plan: Slack RAG Chatbot

**Branch**: `005-rag-chatbot` | **Date**: 2025-12-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-rag-chatbot/spec.md`

## Summary

Slack RAG 챗봇 서비스 - 벡터DB 검색 및 Claude LLM을 통한 Slack 응답 시스템.
사용자가 Slack에서 챗봇을 멘션하거나 DM으로 질문하면, rag-indexer가 인덱싱한 문서를 검색하여 Claude LLM을 통해 답변을 생성합니다.

## Technical Context

**Language/Version**: Python 3.10+ (타입 힌팅 필수)
**Primary Dependencies**:
- slack-bolt>=1.18.0 (Slack Bot Framework)
- anthropic>=0.40.0 (Claude LLM API)
- rag-shared (공유 모듈 - 임베딩, 벡터 스토어)
- redis>=5.0.0 (대화 컨텍스트 및 피드백 저장)
- pydantic>=2.0.0 (데이터 검증)

**Storage**:
- Qdrant (벡터 검색 - shared 모듈 활용)
- Redis (대화 컨텍스트 TTL 저장, 피드백 수집)

**Testing**: pytest, pytest-asyncio, pytest-cov
**Target Platform**: Linux server (Docker), Local development
**Project Type**: Single project (rag-indexer와 동일한 구조)

**Performance Goals**:
- 응답 시작 10초 이내 (SC-001)
- 동시 요청 10개 이상 처리 (SC-005)

**Constraints**:
- Slack 메시지 길이 제한 4000자
- Claude API 토큰 제한 고려
- Redis TTL 기반 세션 관리

**Scale/Scope**:
- 소규모 회사 100명 대상
- 일일 질문 100~500건 예상

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### 핵심 원칙 검증

| 원칙 | 상태 | 검증 내용 |
|------|------|----------|
| I. 비용 효율성 우선 | ✅ PASS | Claude API 사용, 응답당 비용 모니터링 필요. 월 $50 이하 목표 |
| II. 응답 시간 보장 | ✅ PASS | 문서 검색 3초 이내, 전체 응답 10초 이내 목표 |
| III. 정확도 우선 | ✅ PASS | 유사도 0.7 임계값, 출처 표시, "모르겠습니다" 응답 지원 |
| IV. 오픈소스 기반 | ✅ PASS | slack-bolt(MIT), Redis(BSD), Qdrant(Apache 2.0) 사용. Claude API는 유일한 상용 LLM |
| V. 단순성 유지 | ✅ PASS | KISS 원칙 준수. 필수 기능만 구현, 불필요한 추상화 배제 |
| VI. 보안 우선 | ✅ PASS | 민감 정보 가드레일, API 키 환경 변수 관리, 로그 익명화 |

### 헌법 위반 사항

**NONE** - 모든 원칙 준수

### 주의 사항

- **비용**: Claude API 호출 최소화 - 캐싱, 프롬프트 압축 적용 필요
- **LLM**: Constitution에서는 OpenAI GPT-4o를 지정하지만, spec에서는 Claude API 사용 명시. 프로젝트 CLAUDE.md에서도 Anthropic Claude API 명시 → **Claude 사용 결정** (프로젝트 요구사항 우선)

## Project Structure

### Documentation (this feature)

```text
specs/005-rag-chatbot/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── slack-events.md  # Slack 이벤트 스키마 정의
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
rag-chatbot/
├── src/
│   ├── __init__.py
│   ├── main.py              # Slack Bolt 앱 진입점
│   ├── config.py            # 환경 설정
│   ├── handlers/            # Slack 이벤트 핸들러
│   │   ├── __init__.py
│   │   ├── mention.py       # 멘션 이벤트 처리
│   │   ├── dm.py            # DM 이벤트 처리
│   │   └── feedback.py      # 리액션 피드백 처리
│   ├── services/            # 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── rag_service.py   # RAG 검색 + LLM 생성
│   │   ├── conversation.py  # 대화 컨텍스트 관리 (Redis)
│   │   └── feedback.py      # 피드백 수집/저장
│   ├── llm/                 # LLM 통합
│   │   ├── __init__.py
│   │   ├── claude_client.py # Claude API 클라이언트
│   │   └── prompts.py       # 프롬프트 템플릿
│   ├── guardrails/          # 안전 가드레일
│   │   ├── __init__.py
│   │   └── sensitive.py     # 민감 정보 감지
│   └── models/              # 데이터 모델
│       ├── __init__.py
│       ├── query.py         # Query 모델
│       ├── response.py      # Response 모델
│       ├── conversation.py  # Conversation 모델
│       └── feedback.py      # Feedback 모델
├── tests/
│   ├── unit/
│   │   ├── test_rag_service.py
│   │   ├── test_conversation.py
│   │   ├── test_claude_client.py
│   │   └── test_guardrails.py
│   └── integration/
│       ├── test_slack_handlers.py
│       └── test_rag_flow.py
├── pyproject.toml
└── .env.example
```

**Structure Decision**: Single project 구조 (rag-indexer와 동일). shared 모듈을 의존성으로 사용하여 임베딩/벡터 스토어 기능 재활용.

## Complexity Tracking

> **NONE** - Constitution Check에 위반 사항 없음
