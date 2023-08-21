import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np

from amazon_sentiment_model.config.core import config
from amazon_sentiment_model.processing.data_manager import load_dataset
from amazon_sentiment_model.processing.features import process_training_data, get_X_y

                        
@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name=config.app_config.training_data_file)
    data = process_training_data(data)
    
    train_split = config.model_config.train_size
    val_split =  config.model_config.valid_size
    
    a, b, df_test =   np.split(data.sample(frac=1, 
                                                    random_state=config.model_config.random_state), 
                                                    [int(train_split*len(data)), int(val_split*len(data))])
    
    X_test,  y_test   = get_X_y(df_test)
    return X_test, y_test
