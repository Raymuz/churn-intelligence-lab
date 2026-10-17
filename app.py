"""Docker Space: FastAPI REST service plus Gradio web UI, powered by trained pipelines."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / 'model_metadata.json').read_text(encoding='utf-8'))['datasets']
MODELS = {key: joblib.load(ROOT / 'models' / f'{key}_churn.joblib') for key in META}

class SinglePayload(BaseModel):
    data: dict

class BatchPayload(BaseModel):
    records: list[dict] = Field(min_length=1, max_length=100)

def validate_input(key, records):
    if key not in MODELS:
        raise ValueError(f'Unknown segment {key!r}; choose {list(MODELS)}')
    if not records or len(records) > 100:
        raise ValueError('Provide 1–100 records per request')
    info = META[key]
    required = info['features']
    X = pd.DataFrame(records)
    absent = [c for c in required if c not in X.columns]
    if absent:
        raise ValueError(f'Missing required features for {key}: {absent}')
    X = X[required].copy()  # Ignore arbitrary PII/extraneous uploaded columns.
    for c in info['numeric']:
        vals = pd.to_numeric(X[c], errors='coerce')
        if vals.isna().any() or not np.isfinite(vals.to_numpy(dtype=float)).all():
            raise ValueError(f'{c} must contain valid finite numeric values; empty inputs are not accepted')
        X[c] = vals
    for c in info['categorical']:
        allowed = info['categorical_choices'][c]
        vals = X[c].astype('string').str.strip()
        if vals.isna().any() or (~vals.isin(allowed)).any():
            raise ValueError(f'{c} must be one of {allowed}')
        X[c] = vals.astype(str)
    return X

def predict_records(key, records):
    frame = validate_input(key, records)
    probs = MODELS[key]['pipeline'].predict_proba(frame)[:, 1]
    cut = float(MODELS[key]['threshold'])
    low_signal = META[key]['deployment_status'] != 'demonstration'
    return [{'row_number': i+1, 'churn_score': round(float(p), 5),
             'prediction': ('Exploratory only — unreliable classifier' if low_signal
                           else ('Likely to churn' if p >= cut else 'Likely to stay')),
             'deployment_status': META[key]['deployment_status'],
             'threshold': cut} for i, p in enumerate(probs)]

api = FastAPI(title='Churn Intelligence Lab API', version='1.0.0',
              description='Demo-only bank and telecom churn inference. Do not submit real customer PII.')

@api.get('/', include_in_schema=False)
def home():
    return RedirectResponse('/ui/')

@api.get('/health')
def health():
    return {'status': 'ok', 'trained_models': list(MODELS), 'version': '1.0.0'}

@api.get('/api/metadata')
def metadata():
    return {name: {'features': info['features'], 'threshold': info['threshold'],
            'selected_model': info['selected_model'], 'test_metrics': info['test_metrics']}
            for name, info in META.items()}

@api.post('/api/predict/{segment}')
def predict_one(segment: str, body: SinglePayload):
    try:
        return predict_records(segment, [body.data])[0]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

@api.post('/api/predict-batch/{segment}')
def predict_many(segment: str, body: BatchPayload):
    try:
        return {'results': predict_records(segment, body.records)}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

def format_prediction(segment, *values):
    fields = META[segment]['features']
    response = predict_records(segment, [dict(zip(fields, values))])[0]
    return (f'### {response["prediction"]}\n\n'
            f'**Churn model score:** {response["churn_score"]:.1%}  '\
            f'\n\n**Decision threshold:** {response["threshold"]:.1%}  '
            f'\n\n*Uncalibrated model output; screening aid, not a guarantee of churn.*'), response

def run_batch(segment, filepath):
    if not filepath:
        raise gr.Error('Upload a CSV file before requesting batch predictions.')
    try:
        raw = pd.read_csv(filepath)
        if len(raw) == 0 or len(raw) > 100:
            raise ValueError('CSV must contain between 1 and 100 rows')
        output = predict_records(segment, raw.to_dict(orient='records'))
        result_path = Path(tempfile.mkdtemp(prefix='churn_demo_')) / f'{segment}_prediction_results.csv'
        pd.DataFrame(output).to_csv(result_path, index=False)
        return pd.DataFrame(output), str(result_path)
    except (ValueError, pd.errors.ParserError) as error:
        raise gr.Error(str(error)) from error

SITE_CSS = '''
.gradio-container {max-width: 1240px !important; margin: auto !important;}
#hero {background:linear-gradient(120deg,#0a2342,#123a68 65%,#176a89);border-radius:20px;
       padding:28px 34px;color:white; box-shadow:0 12px 40px rgba(5,22,45,.15);}
#hero h1,#hero p {color:white !important;}
#hero h1 {font-weight:800;letter-spacing:-.03em;font-size:2.2rem;}
#metricline {padding:10px 16px;border:1px solid #dce6ef;border-radius:12px;}
#caution {background:#fff7ed;border:1px solid #fed7aa;padding:12px 16px;border-radius:12px;}
#resultbox {border-left:4px solid #0e7490;padding:8px 16px;min-height:90px;}
'''

with gr.Blocks(title='Churn Intelligence Lab | AI Customer Retention', css=SITE_CSS,
               theme=gr.themes.Soft(primary_hue='blue', neutral_hue='slate')) as demo:
    gr.Markdown('''<div id="hero"><p style="opacity:.8;font-weight:700;letter-spacing:.13em">AI • MACHINE LEARNING • MLOPS</p>
    <h1>Churn Intelligence Lab</h1>
    <p>Explore AI-assisted customer attrition insights across banking and telecom. Compare model evidence, run individual assessments, and export anonymized batch results.</p>
    <p style="font-size:.86em;opacity:.85">Portfolio research demonstration • Do not enter real customer identifiers or sensitive information.</p></div>''')
    with gr.Row():
        gr.Markdown('**Two independently evaluated datasets**  \nTrain/test separation · Reproducible preprocessing · Saved production-style pipelines')
        gr.Markdown('**Developer access**  \n[REST API docs](/docs) · [Model metadata](/api/metadata) · [Service health](/health)')
    for key, info in META.items():
        with gr.Tab('🏦  Banking attrition' if key == 'bank' else '📶  Telecom attrition'):
            met = info['test_metrics']
            with gr.Row():
                gr.Markdown(f'''### {"Banking" if key == "bank" else "Telecommunications"} model
                <div id="metricline"> <b>Selected:</b> {info["selected_model"]} &nbsp; | &nbsp;
                <b>Test ROC AUC:</b> {met["roc_auc"]:.3f} &nbsp; | &nbsp;
                <b>PR AUC:</b> {met["pr_auc"]:.3f} &nbsp; | &nbsp;
                <b>Recall:</b> {met["recall"]:.3f}</div>''')
            if info['deployment_status'] != 'demonstration':
                gr.Markdown('''<div id="caution"><b>⚠ Experimental — low predictive reliability.</b>
                The held-out evaluation is near chance. Scores are exploratory; this model is not suitable for business churn decisions.</div>''')
            else:
                gr.Markdown('**Customer assessment** — review sample defaults, modify customer attributes, and run a prediction.')
            inputs=[]
            with gr.Accordion('Customer profile inputs', open=True):
                with gr.Row():
                    with gr.Column():
                        first=info['features'][::2]
                        for col in first:
                            if col in info['numeric']:
                                component=gr.Number(label=col, value=info['defaults'][col], precision=2)
                            else:
                                component=gr.Dropdown(label=col, choices=info['categorical_choices'][col],value=info['defaults'][col])
                            inputs.append((col,component))
                    with gr.Column():
                        for col in info['features'][1::2]:
                            if col in info['numeric']:
                                component=gr.Number(label=col, value=info['defaults'][col], precision=2)
                            else:
                                component=gr.Dropdown(label=col, choices=info['categorical_choices'][col],value=info['defaults'][col])
                            inputs.append((col,component))
            ordered=[dict(inputs)[col] for col in info['features']]
            run=gr.Button('Analyze customer churn risk',variant='primary',size='lg')
            with gr.Row():
                with gr.Column(scale=2):
                    textout=gr.Markdown('Ready to analyze a synthetic customer profile.',elem_id='resultbox')
                with gr.Column(scale=1):
                    jsonout=gr.JSON(label='Prediction details')
            run.click(fn=lambda *values,segment=key: format_prediction(segment,*values),
                      inputs=ordered,outputs=[textout,jsonout],api_name=False)
            gr.Markdown('### Batch screening (CSV)\nUpload 1–100 rows with the required feature columns. Only anonymized or synthetic data is permitted.')
            with gr.Row():
                csvfile=gr.File(label='Anonymized feature CSV',file_types=['.csv'],type='filepath')
                batch_button=gr.Button('Process CSV batch')
            with gr.Row():
                batch_table=gr.Dataframe(label='Prediction results',interactive=False)
                batch_download=gr.File(label='Download results CSV')
            batch_button.click(fn=lambda f,segment=key: run_batch(segment,f),
                               inputs=csvfile,outputs=[batch_table,batch_download],api_name=False)
    gr.Markdown('''---
    **How to interpret scores:** estimated model probabilities are not calibrated. Thresholds are validation-derived; outcomes are uncertain.
    Telecom is research-only because of weak signal. Evaluate fairness, drift, and real-world fit before any operational use.

    **Architecture:** Python · Scikit-learn · XGBoost · Gradio · FastAPI · Docker · Hugging Face Spaces''')

app = gr.mount_gradio_app(api, demo, path='/ui')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)
