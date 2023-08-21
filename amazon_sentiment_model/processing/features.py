
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from amazon_sentiment_model.config.core import config


def strip_html(text):    
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_punctuations(text):

    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)

    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text


def populate_stop_words():

    stopword_list = nltk.corpus.stopwords.words('english')
    updated_stopword_list = []

    for word in stopword_list:
        if word=='not' or word.endswith("n't"):
            pass
        else:
            updated_stopword_list.append(word)
            
    addl_remove_list = ['mustn', 'doesn', "shouldn", "couldn", "didn", "aren", "haven", "mightn", "needn", "shan", "hasn", "isn", "weren"]

    for item in addl_remove_list:
        updated_stopword_list.remove(item)
    
    return updated_stopword_list


def remove_stopwords_and_lemmatize(text):

    #instantiate Lemmatizer
    lmtizer = WordNetLemmatizer()

     # splitting strings into tokens (list of words)
    tokens = nltk.tokenize.word_tokenize(text)  
    tokens = [token.strip() for token in tokens]

    #remove stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]

    #lemmatize the tokens and create the sentence back
    filtered_text = ' '.join([lmtizer.lemmatize(word, pos ='a') for word in filtered_tokens])
    
    return filtered_text


def create_and_update_sentiment_field(input_df) :

    input_df['Sentiment'] = input_df['Score'].apply(lambda x: 1 if x >= 3 else 0)
    return input_df


def drop_duplicate_data(input_df):

    input_df.drop_duplicates(subset=['Sentiment', 'Text'], keep='last', inplace=True)
    return input_df


def preprocess_text(input_df):

    input_df['Text'] = input_df['Text'].apply(strip_html).apply(remove_punctuations).apply(remove_stopwords_and_lemmatize)
    return input_df


def process_training_data(input_df):
    # warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)
    input_df = create_and_update_sentiment_field(input_df)
    input_df = drop_duplicate_data(input_df)
    input_df = preprocess_text(input_df)
    return input_df


def create_tokenizer(training_text):

    tokenizer = Tokenizer(num_words=config.model_config.num_tokenizer_words)
    tokenizer.fit_on_texts(training_text)
    
    return tokenizer


def get_X_y(input_df):
    return (input_df.Text.values, input_df.Sentiment.values)


def create_and_pad_sequences(tokenizer, data):
    token_data = tokenizer.texts_to_sequences(data)
    pad_token_data = pad_sequences(token_data, 
                                    padding='post', 
                                    maxlen= config.model_config.max_seq_len,
                                    truncating='post')
    return pad_token_data
    
stop_word_list = populate_stop_words()
