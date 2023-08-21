import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import typing as t
import pandas as pd
from tensorflow import keras
from keras.models import save_model
from keras.preprocessing.text import Tokenizer,tokenizer_from_json

from amazon_sentiment_model import __version__ as _version
from amazon_sentiment_model.config.core import TRAINED_MODEL_DIR, config


def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
  
    # # Delete unused fields
    # # work around for not being able to specify empty lists in yaml file
    # len_unused = len(config.model_config.unused_fields)
    # if len_unused > 0:
    #     drop_list = []
    #     col_list = data_frame.columns.tolist()
    #     unused_fields = config.model_config.unused_fields
    #     #only select the columns from unused fieldss that exist in data frame
    #     drop_list = [x for x in unused_fields if x in col_list]
    #     if (len(drop_list) > 0):
    #         data_frame.drop(columns=drop_list, inplace=True)

    # drop unnecessary variables
    data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    return data_frame

def load_dataset(*, file_name: str) -> pd.DataFrame:
    # dataframe = pd.read_csv(filepath_or_buffer=file_name, usecols=config.model_config.csv_fields)
    dataframe = pd.read_csv(filepath_or_buffer=file_name)

    transformed = pre_pipeline_preparation(data_frame=dataframe)
    return transformed


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def load_tokenizer(*, filename: str) -> Tokenizer:
    my_tok = None
    with open(filename, "rb") as fd:
        my_tok = tokenizer_from_json(json.load(fd))

    return my_tok


def save_tokenizer(*, tokenizer_to_save:Tokenizer, file_name: str) -> None:
    with open(file_name, "w") as fd:
        json.dump(tokenizer_to_save.to_json(), fd)
  

def save_model_data(model_to_save, tokenizer):
    for model_file in TRAINED_MODEL_DIR.iterdir():
        model_file.unlink()
    keras.models.save_model(model_to_save, TRAINED_MODEL_DIR/f"{config.app_config.model_save_file}{_version}.keras")
    save_tokenizer(tokenizer_to_save=tokenizer, file_name=TRAINED_MODEL_DIR/f"{config.app_config.tokenizer_save_file}{_version}.json")
    
    
def load_model_data( model_file_name: str, tokenizer_file_name: str):
    model = keras.models.load_model(TRAINED_MODEL_DIR/model_file_name)
    tokenizer = load_tokenizer(filename=TRAINED_MODEL_DIR/tokenizer_file_name)
    return tuple((model, tokenizer))
