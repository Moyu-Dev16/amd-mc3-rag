# AMD AI Academy Challenge - Mini-Challenge 3 RAG Engine
# Mandated base image: rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Install system dependencies for OCR & graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-warm RapidOCR so bundled weights and infer engines are ready offline
RUN python3 -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

COPY app/ /app/

# Evaluation harness directories
RUN mkdir -p /app/corpus /app/output /app/index

CMD ["sleep", "infinity"]
