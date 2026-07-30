.PHONY: help up down logs wait test compare train models-list

help:
	@echo "make up / down / logs / wait"
	@echo "make test                      - base (in rõ model trong output)"
	@echo "make test Q=\"...\"              - 1 query trên base"
	@echo "make test MODEL=query-parser-ft Q=\"...\""
	@echo "make compare Q=\"...\"           - base + LoRA cùng query"
	@echo "make train                     - QLoRA one-shot, xong thoát"
	@echo "make models-list"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

wait:
	@echo "Chờ vLLM /health ..."
	@until curl -sf localhost:8000/health >/dev/null; do \
		printf '.'; sleep 5; \
	done; echo " READY"

# MODEL=... chọn model; mặc định google/gemma-4-e2b-it
test:
	@if [ -n "$(Q)" ]; then \
		python3 test_vllm.py -m "$(or $(MODEL),google/gemma-4-e2b-it)" "$(Q)"; \
	else \
		python3 test_vllm.py -m "$(or $(MODEL),google/gemma-4-e2b-it)"; \
	fi

compare:
	@if [ -n "$(Q)" ]; then \
		python3 test_vllm.py --compare "$(Q)"; \
	else \
		python3 test_vllm.py --compare; \
	fi

# One-shot: build → train → container xoá (--rm). Tự stop vLLM để nhả GPU.
train:
	-docker compose stop
	docker compose -f compose.train.yaml run --rm train

models-list:
	@curl -sf localhost:8000/v1/models | python3 -m json.tool
