FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN ls -la /app/
RUN pip install -r /app/requirements.txt

RUN mkdir /app/datasets/
RUN mkdir /app/datasets/train/
RUN mkdir /app/datasets/val/

COPY ./src/ /app/

CMD ["gunicorn", "-b", "0.0.0.0:6767", "app:app"]
