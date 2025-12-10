.PHONY: help infra-up infra-down infra-logs install-indexer install-chatbot install-all test

help:
	@echo "RAG System Commands"
	@echo "==================="
	@echo "infra-up       : Start Docker infrastructure (Qdrant, Redis)"
	@echo "infra-down     : Stop Docker infrastructure"
	@echo "infra-logs     : View infrastructure logs"
	@echo "install-indexer: Install rag-indexer dependencies"
	@echo "install-chatbot: Install rag-chatbot dependencies"
	@echo "install-all    : Install all dependencies"
	@echo "test           : Run all tests"

# Infrastructure commands
infra-up:
	cd infra/docker && docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@echo "Qdrant: http://localhost:6333/dashboard"
	@echo "Redis: localhost:6379"

infra-down:
	cd infra/docker && docker-compose down

infra-logs:
	cd infra/docker && docker-compose logs -f

infra-status:
	cd infra/docker && docker-compose ps

# Install commands
install-indexer:
	cd rag-indexer && pip install -e ".[dev]"

install-chatbot:
	cd rag-chatbot && pip install -e ".[dev]"

install-shared:
	cd shared && pip install -e ".[dev]"

install-all: install-shared install-indexer install-chatbot

# Test commands
test-indexer:
	cd rag-indexer && pytest -v

test-chatbot:
	cd rag-chatbot && pytest -v

test-shared:
	cd shared && pytest -v

test: test-shared test-indexer test-chatbot

# Lint commands
lint:
	ruff check rag-indexer/src rag-chatbot/src shared/src

lint-fix:
	ruff check --fix rag-indexer/src rag-chatbot/src shared/src
