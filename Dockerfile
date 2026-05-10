FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader \
  punkt \
  punkt_tab \
  averaged_perceptron_tagger \
  averaged_perceptron_tagger_eng \
  universal_tagset

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "pipeline.py"]