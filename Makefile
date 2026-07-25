.PHONY: help build up down logs test restart clean

help:
	@echo "make build   - build Docker image"
	@echo "make up      - chạy API (nền), tải model HF lần đầu"
	@echo "make test    - gửi các query mẫu tới API đang chạy (curl)"
	@echo "make logs    - xem log API"
	@echo "make down    - dừng API"
	@echo "make clean   - dừng và xoá volume cache model"

build:
	docker compose build

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

restart: down up

test:
	@for q in "dt ip 256" "laptop dell i5 16gb" "ss s24 ultra 512" "tai nghe bluetooth" "ao thun nam mau den"; do \
		echo "query: $$q"; \
		curl -s localhost:8000/parse -H 'content-type: application/json' -d "{\"query\":\"$$q\"}"; \
		echo; \
	done

clean:
	docker compose down -v
