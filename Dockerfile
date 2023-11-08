FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN ls -la /app/
RUN pip install -r /app/requirements.txt

COPY ./src/ /app/

CMD ["gunicorn", "-b 0.0.0.0:6767", "-w 2", "--threads=2", "app:app"]
