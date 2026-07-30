.PHONY: help up down logs wait ready test train models-list

help:
	@echo "make up / down / logs"
	@echo "make ready           - wait + warmup GPU (trước khi test/prod)"
	@echo "make test Q=\"...\""
	@echo "make train           - QLoRA → adapter"
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

# wait + 1 request warm (tránh cold 3–6s lần đầu)
ready: wait
	@echo "Warmup query-parser-ft ..."
	@python3 test_vllm.py --no-wait --no-warmup -m query-parser-ft "__warmup__" >/dev/null \
		|| { echo "Warmup fail — đã train adapter chưa? models/adapters/query-parser-ft/"; exit 1; }
	@echo " WARM"

test:
	@if [ ! -f models/adapters/query-parser-ft/adapter_config.json ]; then \
		echo "Thiếu adapter. Chạy: make train"; exit 1; \
	fi
	@if [ -n "$(Q)" ]; then \
		python3 test_vllm.py -m "$(or $(MODEL),query-parser-ft)" "$(Q)"; \
	else \
		python3 test_vllm.py -m "$(or $(MODEL),query-parser-ft)"; \
	fi

train:
	@echo "Stopping vLLM / freeing GPU..."
	-docker compose stop
	@sleep 2
	@nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
	docker compose -f compose.train.yaml run --rm train
	@echo "Tiếp: make up && make ready && make test Q=\"dt ip 256\""

models-list:
	@if ! curl -sf localhost:8000/health >/dev/null; then \
		echo "vLLM chưa chạy.  make up && make ready"; \
		exit 1; \
	fi
	@curl -sf localhost:8000/v1/models | python3 -m json.tool
