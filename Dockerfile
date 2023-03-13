FROM python:3.9.16-slim
# -alpine3.17 as base_image
WORKDIR /usr/src/app

COPY ./src/infer_basic/requirements.txt .
#required for uwisgi
RUN apt update
RUN apt install gcc -y

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -m spacy download en_core_web_sm

ENV MODEL_VERSION='roberta'

COPY ./src/infer_basic .

EXPOSE 8080

# CMD ["python3", "main.py"]
CMD ["uwsgi", "--http", "127.0.0.1:8080" "--wsgi-file" "main.py" "--callable" "app", "--threads", "2"]