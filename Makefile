.PHONY: help infra-up infra-down infra-logs infra-status infra-reset \
        install-shared install-indexer install-chatbot install-all \
        test test-shared test-indexer test-chatbot test-cov \
        lint lint-fix format \
        run-chatbot run-chatbot-bg stop-chatbot run-indexer \
        clean clean-cache clean-venv clean-all \
        setup setup-indexer setup-chatbot \
        check health

# 색상 정의
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
RED    := \033[0;31m
NC     := \033[0m # No Color

##@ 도움말
help: ## 사용 가능한 명령어 목록 표시
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║          RAG System - Makefile Commands                  ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ 인프라 관리
infra-up: ## Docker 인프라 시작 (Qdrant + Redis)
	@echo "$(BLUE)▶ Docker 인프라 시작 중...$(NC)"
	cd infra/docker && docker compose up -d
	@echo "$(YELLOW)⏳ 서비스 헬스체크 대기 중...$(NC)"
	@sleep 5
	@echo "$(GREEN)✓ 인프라 시작 완료$(NC)"
	@echo "  • Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo "  • Redis: localhost:6379"

infra-down: ## Docker 인프라 중지
	@echo "$(BLUE)▶ Docker 인프라 중지 중...$(NC)"
	cd infra/docker && docker compose down
	@echo "$(GREEN)✓ 인프라 중지 완료$(NC)"

infra-logs: ## Docker 로그 실시간 확인
	cd infra/docker && docker compose logs -f

infra-status: ## Docker 컨테이너 상태 확인
	@echo "$(BLUE)▶ 인프라 상태$(NC)"
	@cd infra/docker && docker compose ps

infra-reset: ## Docker 인프라 초기화 (볼륨 포함 삭제)
	@echo "$(RED)⚠ 주의: 모든 데이터가 삭제됩니다!$(NC)"
	@read -p "계속하시겠습니까? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	cd infra/docker && docker compose down -v
	@echo "$(GREEN)✓ 인프라 초기화 완료$(NC)"

##@ 설치
install-shared: ## shared 모듈 설치
	@echo "$(BLUE)▶ shared 모듈 설치 중...$(NC)"
	cd shared && pip install -e ".[dev]"
	@echo "$(GREEN)✓ shared 설치 완료$(NC)"

install-indexer: install-shared ## rag-indexer 설치 (shared 포함)
	@echo "$(BLUE)▶ rag-indexer 설치 중...$(NC)"
	cd rag-indexer && pip install -e ".[dev]"
	@echo "$(GREEN)✓ rag-indexer 설치 완료$(NC)"

install-chatbot: install-shared ## rag-chatbot 설치 (shared 포함)
	@echo "$(BLUE)▶ rag-chatbot 설치 중...$(NC)"
	cd rag-chatbot && pip install -e ".[dev]"
	@echo "$(GREEN)✓ rag-chatbot 설치 완료$(NC)"

install-all: install-shared install-indexer install-chatbot ## 모든 모듈 설치
	@echo "$(GREEN)✓ 모든 모듈 설치 완료$(NC)"

##@ 초기 설정
setup: ## 전체 프로젝트 초기 설정 (가상환경 + 의존성 + 인프라)
	@echo "$(BLUE)▶ 전체 프로젝트 설정 시작...$(NC)"
	@echo ""
	@echo "$(YELLOW)1/4 환경 변수 확인$(NC)"
	@if [ ! -f infra/docker/.env ]; then \
		cp infra/docker/.env.example infra/docker/.env; \
		echo "$(YELLOW)  ⚠ .env 파일이 생성되었습니다. API 키를 설정해주세요!$(NC)"; \
		echo "  → infra/docker/.env 파일을 편집하세요."; \
	else \
		echo "$(GREEN)  ✓ .env 파일 존재$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)2/4 의존성 설치$(NC)"
	$(MAKE) install-all
	@echo ""
	@echo "$(YELLOW)3/4 인프라 시작$(NC)"
	$(MAKE) infra-up
	@echo ""
	@echo "$(YELLOW)4/4 헬스체크$(NC)"
	$(MAKE) health
	@echo ""
	@echo "$(GREEN)╔══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║              🎉 설정 완료!                               ║$(NC)"
	@echo "$(GREEN)╚══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "다음 단계:"
	@echo "  1. infra/docker/.env 파일에 API 키 설정"
	@echo "  2. make run-indexer  # 문서 인덱싱"
	@echo "  3. make run-chatbot  # 챗봇 실행"

setup-indexer: ## rag-indexer 전용 가상환경 설정
	@echo "$(BLUE)▶ rag-indexer 가상환경 설정 중...$(NC)"
	cd rag-indexer && python -m venv .venv
	cd rag-indexer && . .venv/bin/activate && pip install --upgrade pip
	cd rag-indexer && . .venv/bin/activate && pip install -e ../shared
	cd rag-indexer && . .venv/bin/activate && pip install -e ".[dev]"
	@echo "$(GREEN)✓ rag-indexer 설정 완료$(NC)"
	@echo "  → cd rag-indexer && source .venv/bin/activate"

setup-chatbot: ## rag-chatbot 전용 가상환경 설정
	@echo "$(BLUE)▶ rag-chatbot 가상환경 설정 중...$(NC)"
	cd rag-chatbot && python -m venv .venv
	cd rag-chatbot && . .venv/bin/activate && pip install --upgrade pip
	cd rag-chatbot && . .venv/bin/activate && pip install -e ../shared
	cd rag-chatbot && . .venv/bin/activate && pip install -e ".[dev]"
	@echo "$(GREEN)✓ rag-chatbot 설정 완료$(NC)"
	@echo "  → cd rag-chatbot && source .venv/bin/activate"

##@ 실행
run-chatbot: ## Slack 챗봇 실행
	@echo "$(BLUE)▶ Slack RAG 챗봇 시작...$(NC)"
	@# .env 심볼릭 링크 확인
	@if [ ! -f rag-chatbot/.env ]; then \
		ln -sf ../infra/docker/.env rag-chatbot/.env; \
		echo "$(YELLOW)  → .env 심볼릭 링크 생성$(NC)"; \
	fi
	@# 가상환경에서 실행
	@if [ -f rag-chatbot/.venv/bin/activate ]; then \
		cd rag-chatbot && . .venv/bin/activate && python -m src.main; \
	else \
		cd rag-chatbot && python -m src.main; \
	fi

run-chatbot-bg: ## Slack 챗봇 백그라운드 실행
	@echo "$(BLUE)▶ Slack RAG 챗봇 백그라운드 시작...$(NC)"
	@# .env 심볼릭 링크 확인
	@if [ ! -f rag-chatbot/.env ]; then \
		ln -sf ../infra/docker/.env rag-chatbot/.env; \
	fi
	@cd rag-chatbot && . .venv/bin/activate && nohup python -m src.main > chatbot.log 2>&1 &
	@sleep 2
	@if pgrep -f "python -m src.main" > /dev/null; then \
		echo "$(GREEN)✓ 챗봇 실행 중 (PID: $$(pgrep -f 'python -m src.main'))$(NC)"; \
		echo "  로그: tail -f rag-chatbot/chatbot.log"; \
	else \
		echo "$(RED)✗ 챗봇 시작 실패$(NC)"; \
		tail -5 rag-chatbot/chatbot.log; \
	fi

stop-chatbot: ## Slack 챗봇 중지
	@echo "$(BLUE)▶ Slack RAG 챗봇 중지...$(NC)"
	@if pgrep -f "python -m src.main" > /dev/null; then \
		pkill -f "python -m src.main"; \
		echo "$(GREEN)✓ 챗봇 중지 완료$(NC)"; \
	else \
		echo "$(YELLOW)⚠ 실행 중인 챗봇 없음$(NC)"; \
	fi

run-indexer: ## Indexer CLI 도움말 표시
	@echo "$(BLUE)▶ RAG Indexer CLI$(NC)"
	@echo ""
	@echo "사용 예시:"
	@echo "  $(GREEN)# Notion 문서 인덱싱$(NC)"
	@echo "  cd rag-indexer && python -m src.cli index-notion --database-id <DB_ID>"
	@echo ""
	@echo "  $(GREEN)# Swagger 인덱싱$(NC)"
	@echo "  cd rag-indexer && python -m src.cli index-swagger --url <SWAGGER_URL>"
	@echo ""
	@echo "  $(GREEN)# 스케줄러 실행$(NC)"
	@echo "  cd rag-indexer && python -m src.cli scheduler"

##@ 테스트
test: test-shared test-indexer test-chatbot ## 모든 테스트 실행
	@echo "$(GREEN)✓ 모든 테스트 완료$(NC)"

test-shared: ## shared 모듈 테스트
	@echo "$(BLUE)▶ shared 테스트 실행...$(NC)"
	cd shared && pytest -v

test-indexer: ## rag-indexer 테스트
	@echo "$(BLUE)▶ rag-indexer 테스트 실행...$(NC)"
	cd rag-indexer && pytest -v

test-chatbot: ## rag-chatbot 테스트
	@echo "$(BLUE)▶ rag-chatbot 테스트 실행...$(NC)"
	cd rag-chatbot && pytest -v

test-cov: ## 커버리지 포함 테스트
	@echo "$(BLUE)▶ 커버리지 테스트 실행...$(NC)"
	cd shared && pytest --cov=shared --cov-report=term-missing
	cd rag-indexer && pytest --cov=src --cov-report=term-missing
	cd rag-chatbot && pytest --cov=src --cov-report=term-missing

##@ 코드 품질
lint: ## 코드 스타일 검사 (Ruff)
	@echo "$(BLUE)▶ 린트 검사 중...$(NC)"
	ruff check shared/shared rag-indexer/src rag-chatbot/src
	@echo "$(GREEN)✓ 린트 검사 완료$(NC)"

lint-fix: ## 자동 수정 가능한 린트 오류 수정
	@echo "$(BLUE)▶ 린트 자동 수정 중...$(NC)"
	ruff check --fix shared/shared rag-indexer/src rag-chatbot/src
	@echo "$(GREEN)✓ 린트 수정 완료$(NC)"

format: ## 코드 포맷팅 (Ruff)
	@echo "$(BLUE)▶ 코드 포맷팅 중...$(NC)"
	ruff format shared/shared rag-indexer/src rag-chatbot/src
	@echo "$(GREEN)✓ 포맷팅 완료$(NC)"

check: lint test ## 린트 + 테스트 전체 검사
	@echo "$(GREEN)✓ 모든 검사 완료$(NC)"

##@ 헬스체크
health: ## 서비스 헬스체크
	@echo "$(BLUE)▶ 서비스 헬스체크$(NC)"
	@echo ""
	@echo "Qdrant:"
	@curl -s http://localhost:6333/health > /dev/null 2>&1 && \
		echo "  $(GREEN)✓ 정상 (http://localhost:6333)$(NC)" || \
		echo "  $(RED)✗ 연결 실패$(NC)"
	@echo ""
	@echo "Redis:"
	@redis-cli ping > /dev/null 2>&1 && \
		echo "  $(GREEN)✓ 정상 (localhost:6379)$(NC)" || \
		echo "  $(YELLOW)⚠ 연결 실패 (redis-cli 필요)$(NC)"

##@ 정리
clean-cache: ## Python 캐시 파일 삭제
	@echo "$(BLUE)▶ 캐시 파일 삭제 중...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ 캐시 삭제 완료$(NC)"

clean-venv: ## 가상환경 삭제
	@echo "$(BLUE)▶ 가상환경 삭제 중...$(NC)"
	rm -rf shared/.venv rag-indexer/.venv rag-chatbot/.venv
	@echo "$(GREEN)✓ 가상환경 삭제 완료$(NC)"

clean-all: clean-cache clean-venv ## 모든 생성 파일 삭제
	@echo "$(GREEN)✓ 전체 정리 완료$(NC)"
