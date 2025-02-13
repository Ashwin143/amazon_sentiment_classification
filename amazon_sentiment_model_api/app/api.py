
import json
import numpy as np
import pandas as pd
from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from amazon_sentiment_model.config.core import config
from amazon_sentiment_model import __version__ as _version
from amazon_sentiment_model.predict import make_prediction

from app.config import settings
from app import __version__, schemas

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Product Review Sentiment prediction with the amazon_sentiment_model
    """
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))
    if "errors" in results and results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results
