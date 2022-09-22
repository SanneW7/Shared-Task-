import pandas as pd
import statistics as stat
from nltk.tokenize import RegexpTokenizer


def pprint_label_distribution(label_distribution):
    """Pretty-prints a label distribution"""
    tot_entries = label_distribution.sum()
    for label_info in label_distribution.iteritems():
        label = label_info[0]
        count = label_info[1]
        pct = '{:.2%}'.format(count / tot_entries)
        print('\"{0}\": {1} ({2})'.format(label, count, pct))


def get_label_distribution(df, column):
    """Return a Series containing counts of a label in a DataFrame column"""
    return df[column].value_counts()

def get_avg_length(df, column):
    tokenizer = RegexpTokenizer(r'\w+')
    texts = df[column].to_list()
    text_lengths = [len(tokenizer.tokenize(text)) for text in texts]
    avg_length = stat.mean(text_lengths)
    return avg_length



def main():
    # Load training data
    train_path = "starting_kit/train_all_tasks.csv"
    train_df = pd.read_csv(train_path)

    # Get label distribution of the column 'label_sexist' in training data
    label_dist_a = get_label_distribution(train_df, 'label_sexist')
    pprint_label_distribution(label_dist_a)

    # Average length of texts in training data in words
    length_w_a = get_avg_length(train_df, 'text')
    print(length_w_a)


if __name__ == "__main__":
    main()
