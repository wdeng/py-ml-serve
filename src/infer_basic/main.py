# https://github.com/microsoft/onnxruntime/issues/11156
import logging
import os
import time
import json

from flask import Flask, jsonify, request, Response
from models import get_qa_predictor
# from prometheus_client import Histogram, generate_latest

__version__ = os.getenv('MODEL_VERSION', 'roberta')

# endpoint_latency = Histogram(
#     "endpoint_latency_seconds",
#     f"Latency for endpoints {MODEL_VERSION} model",
#     ["endpoint"],
# )

logging.basicConfig(
  filename='app.log',
  level=logging.INFO,
  format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logging.info("Setting LOGLEVEL to INFO")

# Create my app
app = Flask(__name__)


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
    # endpoint_latency.labels(request.path).observe(latency)

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


@app.route("/<string:qa_type>", methods=['POST'])
def question_answer_reqs(qa_type):
    result, err_status = {}, 404

    predictor = get_qa_predictor()
    if qa_type not in [
        'question-answer', 'clean-text',
        'sentencize-text', 'clean-and-sentencize'
    ]:
        err = "the api endpoint is not supported, please only use question-answer, clean-text, sentencize-text, or clean-and-sentencize"
        return jsonify(error=err), err_status
    logger = logging.getLogger(qa_type)
    # parse json input
    try:
        payload = request.json
    except Exception:
        err = f'cannot parse the json payload {request.data}'
        logger.warning(err)
        return jsonify(error=err), err_status

    if qa_type == 'question-answer':
        # for the question-answer endpoint
        if not payload.get('text') or not isinstance(payload.get('question'), list):
            err = 'text preprocessing needs "text" and "question" field in the json request'
            logger.warning(err)
            return jsonify(error=err), err_status
        try:
            result = []
            text = payload['text']
            for question in payload['question']:
                t0 = time.time()
                res = predictor.question_answering(question, text)
                res['metadata'] = {'compute_time': time.time() - t0}
                result.append()
        except Exception as e:
            err = f'cannot finish the processing task {qa_type}: {e}'
            logger.error(err)
            return jsonify(error=err), err_status
    else:
        # for other text processing endpoints
        if not payload.get('text'):
            err = 'text preprocessing needs "text" field in the json request'
            logger.warning(err)
            return jsonify(error=err), err_status
        try:
            text = payload['text']
            if qa_type == 'clean-text':
                result = predictor.clean_text(text)
            elif qa_type == 'sentencize-text':
                result = predictor.sentencize(text)
            else:
                result = predictor.clean_text(text)
                result = predictor.sentencize(result)
        except Exception as e:
            err = f'cannot finish the processing task {qa_type}: {e}'
            logger.error(err)
            return jsonify(error=err), err_status
        result = {'text': result}

    return jsonify(result)


@app.route("/metrics")
def metrics():
    """
    For Prometheus monitoring metrics
    """
    # return Response(generate_latest(REGISTRY), mimetype='text/plain')
    return Response('hi', mimetype='text/plain')


@app.route("/health")
def health():
    """
    For k8s health check
    """
    return 'OK'


if __name__ == "__main__":
    app.run()
