FROM python:3.9.16-slim
# -alpine3.17 as base_image
WORKDIR /usr/src/app

COPY ./src/infer/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./src/infer .

EXPOSE 5000

CMD ["python3", "main.py"]
