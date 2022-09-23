import pandas as pd
import statistics as stat
from nltk.tokenize import RegexpTokenizer
#pip install emoji
import emoji


def pprint_label_distribution(label_distribution):
    """Pretty-prints a label distribution"""
    tot_entries = label_distribution.sum()
    for label_info in label_distribution.iteritems():
        label = label_info[0]
        count = label_info[1]
        pct = '{:.2%}'.format(count / tot_entries)
        print('\"{0}\": {1} ({2})'.format(label, count, pct))
    print("\n")


def get_label_distribution(df, column):
    """Return a Series containing counts of a label in a DataFrame column"""
    return df[column].value_counts()


def get_avg_length(df, column):
    tokenizer = RegexpTokenizer(r'\w+')
    texts = df[column].to_list()
    text_lengths = [len(tokenizer.tokenize(text)) for text in texts]
    avg_length = stat.mean(text_lengths)
    return avg_length


def contains_emoji(s):
    # tering langzaam maar werkt ouleh
    count = 0
    for e in emoji.EMOJI_DATA:
        count += s.count(e)
        if count > 1:
            return False
    return bool(count)
    

def sents_containing_emoji(df, column):
    df['contains_emoji'] = df[column].apply(lambda e: contains_emoji(e))
    dist = df['contains_emoji'].value_counts()
    
    return dist


def main():
    # Load training data
    train_path = "starting_kit/train_all_tasks.csv"
    train_df = pd.read_csv(train_path)

    # Get label distribution of the column 'label_sexist' in training data
    label_dist_a = get_label_distribution(train_df, 'label_sexist')
    print("label distribution sexist/not sexist")
    pprint_label_distribution(label_dist_a)

    # Average length of texts in training data in words
    length_w_a = get_avg_length(train_df, 'text')
    print("average word length:", length_w_a, "\n")

    # label distribution of label_category
    dist = get_label_distribution(train_df, 'label_category')
    print("label distribution label categories")
    pprint_label_distribution(dist)

    # label distribution of label_vector
    dist_v = get_label_distribution(train_df, 'label_vector')
    print("label distruibution label_vector")
    pprint_label_distribution(dist_v)

    #get sents containing emojis
    dist_e = sents_containing_emoji(train_df, 'text')
    print("sentences containing emojis")
    pprint_label_distribution(dist_e)

    


if __name__ == "__main__":
    main()