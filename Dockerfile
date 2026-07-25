FROM python:3.12-slim

WORKDIR /app

COPY app/requirements.txt .

# torch bản CUDA (cu124) để dùng GPU NVIDIA trên server.
# Wheel này đã bundle sẵn CUDA runtime; host chỉ cần NVIDIA driver +
# nvidia-container-toolkit. Đổi cu124 -> cu121/cu118 nếu driver cũ hơn.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
