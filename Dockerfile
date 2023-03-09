FROM python:3.9.16-alpine3.17 as base_image
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./src .

EXPOSE 5000

CMD ["python3"]
