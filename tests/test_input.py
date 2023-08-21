
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import string
import numpy as np
from random import randint
from amazon_sentiment_model import __version__ as _version
from amazon_sentiment_model.config.core import DATASET_DIR, TRAINED_TOKEN_DIR, config
from amazon_sentiment_model.processing.data_manager import load_tokenizer


def test_input_data(sample_input_data):
    # Given
    X_test = sample_input_data[0]
    y_label = sample_input_data[1]
   

    test_len = len(X_test)
    result_list = []
    #Make sure there are no punctuation chars in random test strings
    for i in range(0,5):
        num = randint(0, test_len)
        if not any(p in X_test[num] for p in string.punctuation):
            result_list.append(False)
        else :
            result_list.append(True)

    assert  True not in result_list 
    
    #Verify that input data contains only 2 values for sentiment values
    print(np.unique(y_label,axis=0,return_counts=True))
    assert len(np.unique(y_label,axis=0,return_counts=True)[0]) == 2
    #verify that Tokenizer can be loaded and is not null
    tokenizer_file_name = f"{config.app_config.tokenizer_save_file}{_version}.json"
    tokenizer = load_tokenizer(filename=TRAINED_TOKEN_DIR/tokenizer_file_name)
    assert tokenizer != None
    
