.PHONY: help up down logs wait test restart

help:
	@echo "make up      - chạy vLLM"
	@echo "make wait    - chờ model sẵn sàng"
	@echo "make test    - gọi trực tiếp API vLLM + JSON schema"
	@echo "make logs    - xem log"
	@echo "make down    - dừng"

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

test:
	python3 test_vllm.py

restart: down up
