
from pydantic import BaseModel
from typing import Any, List, Optional
from amazon_sentiment_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[list[bool]]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Id": 1,
                        "ProductId":  'B001E4KFG0',
                        "UserId": 'A3SGXH7AUHU8GW',
                        "ProfileName": 'delmartian',
                        "HelpfulnessNumerator": 1,
                        "HelpfulnessDenominator": 1,
                        "Score": 5,
                        "Time": 1303862400,
                        "Summary": 'Good Quality Dog Food',
                        "Text": 'I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.',
                    }
                ]
            }
        }
