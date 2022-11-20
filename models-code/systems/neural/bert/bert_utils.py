from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

import sys
sys.path.append("../")
from utils import read_corpus, filter_none_class

def load_model(lm = "bert-base-uncased", num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels= num_labels)
    return model, tokenizer

def vectorize_inputtext(max_seq_len, tokenizer, listoftexts):
    # Transform words to indices using a vectorizer
    vectorized_tokens = tokenizer(listoftexts, padding=True, max_length=max_seq_len,
    truncation=True, return_tensors="np").data
    return vectorized_tokens

def read_testdata_andvectorize(test_filename, max_seq_len, tokenizer, encoder, task_type):
    # Read in test set and vectorize
    X_test, Y_test = read_corpus(test_filename, ",", task_type)
    if task_type != "A":
        X_test, Y_test = filter_none_class(X_test, Y_test)

    tokens_test = vectorize_inputtext(max_seq_len, tokenizer, X_test)
    Y_test_bin = encoder.fit_transform(Y_test)
    return X_test, Y_test, tokens_test, Y_test_bin
