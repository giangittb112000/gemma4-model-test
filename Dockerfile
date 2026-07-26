# torch 2.5.1 + CUDA 12.4 + cuDNN 9 đã cài sẵn, khớp phiên bản (giống các
# dự án GPU khác). Không cần pip install torch nữa -> build nhanh & ổn định.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
