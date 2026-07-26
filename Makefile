.PHONY: help build up down logs test restart clean pull wait

help:
	@echo "make pull    - tải model về ./hf-cache 1 lần (không cần start server)"
	@echo "make up      - chạy vllm + api (nền)"
	@echo "make wait    - chờ tới khi model load xong (/health = ok)"
	@echo "make test    - gửi các query mẫu tới api đang chạy (curl)"
	@echo "make logs    - xem log (vllm + api)"
	@echo "make down    - dừng"
	@echo "make clean   - dừng (KHÔNG xoá ./hf-cache; xoá cache thì rm -rf ./hf-cache)"

build:
	docker compose build

pull:
	docker compose run --rm --no-deps --entrypoint hf vllm \
		download google/gemma-4-e2b-it

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

restart: down up

wait:
	@echo "Chờ model load xong..."
	@until curl -sf localhost:8000/health | grep -q '"status":"ok"'; do \
		printf '.'; sleep 5; \
	done; echo " READY"

test:
	@for q in "dt ip 256" "laptop dell i5 16gb" "ss s24 ultra 512" "tai nghe bluetooth" "ao thun nam mau den"; do \
		echo "query: $$q"; \
		curl -s localhost:8000/parse -H 'content-type: application/json' -d "{\"query\":\"$$q\"}"; \
		echo; \
	done

clean:
	docker compose down
