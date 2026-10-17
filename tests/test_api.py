"""Run after training: python -m pytest -q tests/"""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from app import app, META, predict_records

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert set(r.json()['trained_models']) == {'bank', 'telecom'}

def test_ui_redirect():
    r = client.get('/', follow_redirects=False)
    assert r.status_code in (302, 307, 308)
    assert r.headers['location'] == '/ui/'

def test_predictions():
    for segment, info in META.items():
        sample = info['defaults']
        r = client.post(f'/api/predict/{segment}', json={'data': sample})
        assert r.status_code == 200, r.text
        result = r.json()
        assert 0 <= result['churn_score'] <= 1
        assert result['prediction'] in ('Likely to churn', 'Likely to stay',
                                       'Exploratory only — unreliable classifier')
        if segment == 'telecom' and info['deployment_status'] != 'demonstration':
            assert result['prediction'].startswith('Exploratory only')
        r_batch = client.post(f'/api/predict-batch/{segment}', json={'records': [sample, sample]})
        assert r_batch.status_code == 200, r_batch.text
        assert len(r_batch.json()['results']) == 2
        assert r_batch.json()['results'][0]['churn_score'] == result['churn_score']

def test_invalid_fields_rejected():
    r = client.post('/api/predict/bank', json={'data': {'Age': 40}})
    assert r.status_code == 422
    r2 = client.post('/api/predict/not-a-model', json={'data': META['bank']['defaults']})
    assert r2.status_code == 422

def test_no_identifiers_in_model_schema():
    for name, info in META.items():
        assert not set(info['features']) & set(info['excluded_columns'])
        assert len(info['features']) >= 9


def test_swagger_documentation():
    response = client.get('/openapi.json')
    assert response.status_code == 200
    assert '/api/predict/bank' not in response.json()['paths']  # routing uses path template
    assert '/api/predict/{segment}' in response.json()['paths']

def test_unexpected_category_or_nonfinite_rejected():
    bank = dict(META['bank']['defaults'])
    bank['Geography'] = 'UNLISTED'
    assert client.post('/api/predict/bank', json={'data':bank}).status_code == 422
    bank = dict(META['bank']['defaults'])
    bank['Age'] = 'not-a-number'
    assert client.post('/api/predict/bank', json={'data':bank}).status_code == 422

def test_batch_above_limit_rejected():
    payload = {'records':[dict(META['bank']['defaults']) for _ in range(101)]}
    assert client.post('/api/predict-batch/bank', json=payload).status_code == 422
