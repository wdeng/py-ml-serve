# https://github.com/microsoft/onnxruntime/issues/11156
import logging
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app
# TODO:
# jupyter test
# eval accuracy, time
# pytest
# logging
# prometheus monitoring
# exception handling


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")
logger.info("Setting LOGLEVEL to INFO")

# Create my app
app = Flask(__name__)

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})


@app.route("/hello")
def hello():
    return "hello"


@app.route("/")
def main():
    return "init"


if __name__ == "__main__":
    app.run(host='0.0.0.0')

# bash
# flask --app main run # basic

# uwsgi
# https://flask.palletsprojects.com/en/2.2.x/deploying/uwsgi/
# https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-uswgi-and-nginx-on-ubuntu-18-04
# uwsgi --http 127.0.0.1:8080 --wsgi-file main.py --callable app
# uwsgi --socket 0.0.0.0:8080 --protocol=http -w main:app
