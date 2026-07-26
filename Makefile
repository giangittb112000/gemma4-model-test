.PHONY: help up down logs wait test restart

help:
	@echo "make up              - chạy vLLM"
	@echo "make wait            - chờ model sẵn sàng"
	@echo "make test            - chạy bộ query mặc định"
	@echo "make test Q=\"...\"    - test 1 query tùy chọn"
	@echo "make logs            - xem log"
	@echo "make down            - dừng"

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

# Ví dụ:
#   make test
#   make test Q="dt ip 256"
#   make test Q="ss s24 ultra"
test:
	@if [ -n "$(Q)" ]; then \
		python3 test_vllm.py "$(Q)"; \
	else \
		python3 test_vllm.py; \
	fi

restart: down up
