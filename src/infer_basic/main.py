import logging
import os
import time
import json

from flask import Flask, jsonify, request, Response
from models import get_qa_predictor
from monitor import CustomMetrics
from prometheus_client import generate_latest, REGISTRY

__version__ = os.getenv('MODEL_VERSION', 'roberta')

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logging.info("Setting LOGLEVEL to INFO")

# Create my app
qa_predictor = get_qa_predictor()
app = Flask(__name__)
# Setup Prometheus monitoring metrics
prom_metrics = CustomMetrics()
# REGISTRY.register(prom_metrics)


@app.before_request
def before_request():
    request.start_time = time.time()


@app.after_request
def after_request(response):
    if request.endpoint in ['metrics', 'health']:
        return response
    data = response.get_json()
    _time = time.time() - request.start_time

    # add metrics for Prometheus
    prom_metrics.label(request, response, _time)

    # add metadata to all the request
    payload = {
        'compute_time': _time,
        'model_version': __version__,
        'status_code': response.status_code
    }
    if isinstance(data, dict):
        data['metadata'] = {**payload, **data.get('metadata', {})}
    elif isinstance(data, list):
        for elem in data:
            elem['metadata'] = {**payload, **elem.get('metadata', {})}

    response.data = json.dumps(data)

    return response


@app.route("/clean-and-sentencize", methods=['POST'])
@app.route("/clean-text", endpoint='clean-text', methods=['POST'])
@app.route("/sentencize-text", endpoint='sentencize-text', methods=['POST'])
def preprocess():
    err_status = 400
    qa_type = request.endpoint
    logger = logging.getLogger(qa_type)
    # parse json input
    print(request)
    try:
        payload = request.json
        text = payload['text']
    except Exception:
        err = f'cannot parse the json payload: {request.data}'
        logger.warning(err)
        return jsonify(error=err), err_status

    # for other text processing endpoints
    try:
        if qa_type == 'clean-text':
            result = qa_predictor.clean_text(text)
        elif qa_type == 'sentencize-text':
            result = qa_predictor.sentencize(text)
        else:
            result = qa_predictor.clean_text(text)
            result = qa_predictor.sentencize(result)
    except Exception as e:
        err = f'cannot finish the processing task {qa_type}: {e}'
        logger.error(err)
        return jsonify(error=err), 500
    logger.info(f'{qa_type} request is finished.')
    return jsonify(text=result)


@app.route("/question-answer", methods=['POST'])
def question_answer():
    err_status = 400

    logger = logging.getLogger('question-answer')
    # parse json input
    try:
        payload = request.json
        text = payload['text']
        questions = payload['question']
    except Exception:
        err = f'cannot parse the json payload: {request.data}'
        logger.warning(err)
        return jsonify(error=err), err_status

    # for the question-answer endpoint
    if not (payload.get('text') and isinstance(payload.get('question'), list)):
        err = 'needs "text" and "question" field in json request'
        logger.warning(err)
        return jsonify(error=err), err_status
    try:
        result = []
        for question in questions:
            t0 = time.time()
            res = qa_predictor.question_answering(question, text)
            res['metadata'] = {'compute_time': time.time() - t0}
            result.append(res)
    except Exception as e:
        err = f'cannot finish the task question-answer: {e}'
        logger.error(err)
        return jsonify(error=err), 500
    logger.info('question answer is finished.')
    return jsonify(result)


@app.route("/metrics")
def metrics():
    """
    For Prometheus monitoring metrics
    """
    return Response(generate_latest(REGISTRY), mimetype='text/plain')


@app.route("/health")
def health():
    """
    For k8s health check
    """
    return 'OK'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080')
# uwsgi --http 127.0.0.1:8080 --wsgi-file main.py --callable app --threads 4
