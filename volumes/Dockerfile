FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./llm-weights/download_models.py .

CMD ["python", "download_models.py"]