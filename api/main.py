from fastapi import FastAPI
from .app.models import PredictionRequest, PredictionResponse
from .app.views import get_prediction


app = FastAPI(docs_url='/')

@app.post('/v1/prediciton')
def make_model_prediciton(request: PredictionRequest):
    return PredictionRequest(worldwide_gross=get_prediction(request))