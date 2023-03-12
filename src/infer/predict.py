# predict.py
# https://www.codingforentrepreneurs.com/blog/onnx-machine-learning-in-production/
import json
import pathlib

import onnxruntime

from .encoding import NumpyEncoder
from .preprocessing import process_image
ONNX_SESSION = None


def get_session():
    global ONNX_SESSION
    if ONNX_SESSION is None:
        model_path = str(pathlib.Path("model.onnx"))
        sess = onnxruntime.InferenceSession(model_path)
        ONNX_SESSION = sess
    return ONNX_SESSION


def predict(img_path, use_array=False, *args, **kwargs):
    onnx_sess = get_session()
    sess_inputs = onnx_sess.get_inputs()[0]
    input_name = sess_inputs.name
    shape = sess_inputs.shape
    im = process_image(img_path, height=shape[1], width=shape[2])
    inference_preds = onnx_sess.run(None, {input_name: im})
    results = inference_preds[0][0]
    data = {str(k): v for k, v in enumerate(results)}
    return json.dumps(data, cls=NumpyEncoder)
