# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load
sys.path.append(str(Path(__file__).parent.parent.parent))
import amazon_sentiment_model

# Project Directories
PACKAGE_ROOT = Path(amazon_sentiment_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    model_save_file: str
    tokenizer_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    
    target: str
    features: List[str]
    unused_fields: List[str]
    epoch_list: List[int]
    train_size:float
    valid_size:float
    random_state: int
    num_tokenizer_words: int
    max_seq_len: int
    embedding_dim: int
    csv_fields: List[str]

 
class Config(BaseModel):
    """Master config object."""
    
    model_config: ModelConfig
    app_config: AppConfig
   

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    model_config = ModelConfig(**parsed_config.data)
    #print(model_config)
    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        model_config=model_config,
        app_config=AppConfig(**parsed_config.data),
        #model_config=model_config
    )

    return _config


config = create_and_validate_config()
