from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModel
from keras import Sequential
from keras.layers import Dense

import sys
import pandas as pd
sys.path.append("../")
from utils import read_corpus, filter_none_class, get_taskname

def load_model(lm = "bert-base-uncased", num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokenizer.add_tokens("[NUM]")
    try:
        auto_model = TFAutoModelForSequenceClassification.from_pretrained(lm)
    except:
        auto_model = TFAutoModelForSequenceClassification.from_pretrained(lm, from_pt=True)
    auto_model.resize_token_embeddings(len(tokenizer))
    # This is the same as before, with no added layers. By using model.summary() in train.py (see train,py),
    # you can see the structure of the loaded model, including a classifier layer.

    # This is what we discussed in the cafetaria, when we got the error of the input size. We asked Tommaso and
    # He recommended extracting the CLS layer. This is not that. You might want to change the variable names.
    # model = Sequential()
    # model.add(auto_model)
    # model.add(Dense(num_labels))
    return auto_model, tokenizer

def vectorize_inputtext(max_seq_len, tokenizer, listoftexts):
    # Transform words to indices using a vectorizer
    vectorized_tokens = tokenizer(listoftexts, padding=True, max_length=max_seq_len,
    truncation=True, return_tensors="np").data
    return vectorized_tokens

def read_testdata_andvectorize(test_filename, max_seq_len, tokenizer, encoder, task_type):
    # Read in test set and vectorize
    test_ids, X_test, Y_test = read_corpus(test_filename, ",", task_type)
    if task_type != "A":
        test_ids, X_test, Y_test = filter_none_class(test_ids, X_test, Y_test)

    tokens_test = vectorize_inputtext(max_seq_len, tokenizer, X_test)
    return test_ids, X_test, Y_test, tokens_test
