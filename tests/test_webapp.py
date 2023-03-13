import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src/infer_basic"))

from main import app


def test_qa():
    with open(Path(__file__).parent / 'qa_payloads.json') as f:
        payloads = json.load(f)
    for payload in payloads:
        resp = app.test_client().post('/question-answer', json=payload)
        assert resp.status_code == 200
        results = resp.json
        assert isinstance(results, list)
        for r in results:
            assert ('answer' in r) and ('score' in r) and ('metadata' in r)


def test_preprocess():
    with open(Path(__file__).parent / 'preprocess_payloads.json') as f:
        payloads = json.load(f)
    for payload in payloads:
        for endpoint in ['clean-text', 'clean-and-sentencize', 'sentencize-text']:
            resp = app.test_client().post(endpoint, json={'text': payload['input']})
            assert resp.status_code == 200
            results = resp.json
            assert 'error' not in results
            assert 'text' in results
            assert results['text'] == payload[f'expected-{endpoint}']
