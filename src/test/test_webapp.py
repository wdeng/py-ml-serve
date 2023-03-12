from infer_basic.main import app
import json


def test_qa():
    payloads = json.load('./qa_payloads.json')
    for data in payloads:
        resp = app.test_client().post('/question-answers', json=data)
        assert resp.status_code == 200


def test_preprocess():
    payloads = json.load('./preprocess_payloads.json')
    for data in payloads:
        for endpoint in ['clean-text', '']:
            resp = app.test_client().post('/', json=data)
            assert resp.status_code == 200
