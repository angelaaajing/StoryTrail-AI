FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./models ./models
COPY ./scripts ./scripts

EXPOSE 7860

CMD ["python", "app/main.py"]
