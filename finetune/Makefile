.PHONY: help train clean

help:
	@echo "make train   - QLoRA train từ data/train.json rồi merge -> outputs/merged"
	@echo "make clean   - xoá outputs/"

train:
	python3 train_qlora.py

clean:
	rm -rf outputs
