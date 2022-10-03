FROM python:3.8-slim-buster

WORKDIR /app

COPY ["req.txt", "app.py", "cifar10_classes.txt", "logs/train/runs/2022-09-30_10-41-11/model.trace.pt", "./"]

RUN pip install --no-cache-dir -r req.txt

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "./app.py", "--server.port",  "8080", "--", "--model", "model.trace.pt"]
