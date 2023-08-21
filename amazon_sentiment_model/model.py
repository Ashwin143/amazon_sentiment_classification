import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from amazon_sentiment_model.config.core import config


'''
Method for creating TF model. Invoked during the 
pipeline and Hyperparameter tuning. Only exposing 2 parameters
as github takes long time for GridSearch
'''
def create_tf_model(word_length):

    vocab_size = word_length + 1
      
    amazon_sentiment_model = Sequential([
        keras.layers.Embedding(input_dim = vocab_size, 
                               output_dim = config.model_config.embedding_dim, 
                               input_length=config.model_config.max_seq_len),
        Bidirectional(LSTM(units=60, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    amazon_sentiment_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return amazon_sentiment_model
