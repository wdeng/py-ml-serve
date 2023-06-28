# compile the models into ONNX
FROM python:3.9.16-slim as base_image
WORKDIR /usr/src/app

COPY ./src/onnx_deploy/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./src/onnx_deploy .
RUN python main.py

# inference model with precompiled model
FROM python:3.9.16-slim

LABEL author="Will Deng"
LABEL author-email="dengwenxiang@gmail.com"

WORKDIR /usr/src/app

# required for uwsigi
RUN apt update && apt install gcc -y

COPY ./src/onnx_infer/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -m spacy download en_core_web_sm

COPY --from=base_image /usr/src/app/qa_model ./qa_model
COPY --from=base_image /usr/src/app/qa_tokenizer ./qa_tokenizer

ENV MODEL_VERSION='roberta_onnx'

COPY ./src/onnx_infer .

EXPOSE 8080

# CMD ["python3", "main.py"]
CMD ["uwsgi", "--http", "127.0.0.1:8080", "--wsgi-file", "main.py", "--callable", "app"]
