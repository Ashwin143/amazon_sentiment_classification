
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from amazon_sentiment_model.config.core import config
from amazon_sentiment_model import __version__ as _version
from amazon_sentiment_model.processing.data_manager import load_model_data
from amazon_sentiment_model.processing.validation import validate_inputs
from amazon_sentiment_model.processing.features import preprocess_text,create_and_pad_sequences


model_file_name = f"{config.app_config.model_save_file}{_version}.keras"
tokenizer_file_name = f"{config.app_config.tokenizer_save_file}{_version}.json"
amazon_sentiment_model, tokenizer = load_model_data(model_file_name=model_file_name,
                                 tokenizer_file_name=tokenizer_file_name)


def make_prediction(*,input_data: dict, return_as_tuple=False) -> dict:
    '''
    Make a prediction using a saved model
    the output format is either np arrray of the target label
    or a string representation of the same
    ''' 
       
    results = {"predictions": None, "version": _version, }
    try :
        validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
        #return results
        if errors:
            results['errors'] = errors
        else:
            
            batch_size = 1 if len(validated_data) < 128 else 128
            
            data = preprocess_text(validated_data)
            sequences = create_and_pad_sequences(tokenizer=tokenizer, data=data.Text.values)
            predictions = amazon_sentiment_model.predict(sequences, batch_size=batch_size)
            
            out_list = []
            for element in predictions:
                out_list.append( element[0] >= 0.5 if return_as_tuple == False else (element[0] >= 0.5, element[0]))
         
            results["predictions"] = out_list
    
        #print(results)
    except Exception as e:
        results['errors'] = str(e)
        
    return results


if __name__ == "__main__":
    
    data_in = {'Id': [1], 
               'ProductId': ['B001E4KFG0'], 
               'UserId': ['A3SGXH7AUHU8GW'], 
               'ProfileName': ['delmatian'], 
               'HelpfulnessNumerator': [1], 
               'HelpfulnessDenominator': [1], 
               'Score': [5], 
               'Time': [1303862400], 
               'Summary': ['Good Quality Dog Food'], 
               'Text': ['I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.']
               }
    
    results = make_prediction(input_data=data_in, return_as_tuple= True)
    print(results)
