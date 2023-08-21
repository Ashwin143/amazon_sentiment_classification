
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import classification_report

from amazon_sentiment_model.config.core import config
from amazon_sentiment_model.model import  create_tf_model
from amazon_sentiment_model.processing.data_manager import save_model_data,load_dataset
from amazon_sentiment_model.processing.features import create_and_pad_sequences, create_tokenizer, get_X_y, process_training_data


def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    data = process_training_data(data)
   
    train_split = config.model_config.train_size
    val_split = config.model_config.valid_size
    
    df_train, df_valid, df_test =   np.split(data.sample(frac=1, 
                                                    random_state=config.model_config.random_state), 
                                                    [int(train_split*len(data)), int(val_split*len(data))])
    
    X_train, y_train  = get_X_y(df_train)
    X_valid, y_valid  = get_X_y(df_valid)
    X_test,  y_test   = get_X_y(df_test)
    
    tokenizer = create_tokenizer(X_train)
    
    X_train_pad = create_and_pad_sequences(tokenizer, X_train)
    X_valid_pad = create_and_pad_sequences(tokenizer, X_valid)
    X_test_pad  = create_and_pad_sequences(tokenizer, X_test)
    
    model = create_tf_model(len(tokenizer.word_index))
    model.fit(X_train_pad, y_train, batch_size=256, epochs=1 , validation_data=(X_valid_pad, y_valid))   
    
    save_model_data(model_to_save=model, tokenizer=tokenizer)    
    y_pred = model.predict(X_test_pad,batch_size=128)
    y_pred_bool = (y_pred>= 0.5)
    print(classification_report(y_test,y_pred_bool)) 
    
    
if __name__ == "__main__":
    run_training()