import pandas as pd
import statistics as stat
from nltk.tokenize import RegexpTokenizer
# pip install emoji
import emoji


def divider():
    """Print a divider for output readability"""
    print('-'*50)

def pprint_label_distribution(label_distribution):
    """Pretty-prints a label distribution"""
    tot_entries = label_distribution.sum()
    for label_info in label_distribution.iteritems():
        label = label_info[0]
        count = label_info[1]
        pct = '{:.2%}'.format(count / tot_entries)
        print('\"{0}\": {1} ({2})'.format(label, count, pct))
    print()


def get_label_distribution(df, column):
    """Return a Series containing counts of a label in a DataFrame column"""
    return df[column].value_counts()


def get_avg_length(df, column):
    """Get average length of a texts in words"""
    tokenizer = RegexpTokenizer(r'\w+')
    texts = df[column].to_list()
    text_lengths = [len(tokenizer.tokenize(text)) for text in texts]
    avg_length = stat.mean(text_lengths)
    return round(avg_length, 2)


def contains_emoji(s):
    """Checks if there is an emoji in a text"""
    count = 0
    for e in emoji.EMOJI_DATA:
        count += s.count(e)
        if count > 1:
            return False
    return bool(count)


def sents_containing_emoji(df, column):
    """Return a Series containing the distribution of emoji-presence in texts"""
    df['contains_emoji'] = df[column].apply(lambda e: contains_emoji(e))
    dist = df['contains_emoji'].value_counts()
    return dist


def main():
    # Load training data
    train_path = "starting_kit/train_all_tasks.csv"
    train_df = pd.read_csv(train_path)
    train_df_sexist = train_df[train_df['label_sexist'] == 'sexist'].reset_index(drop=True)
    train_df_not_sexist = train_df[train_df['label_sexist'] == 'not sexist'].reset_index(drop=True)

    # Get label distribution of the column 'label_sexist' in training data
    label_dist_ls = get_label_distribution(train_df, 'label_sexist')
    print("* Label distribution 'label_sexist' (Task A) *")
    pprint_label_distribution(label_dist_ls)

    # Get label distribution of the column 'label_category' in training data
    label_dist_lc = get_label_distribution(train_df, 'label_category')
    print("* Label distribution 'label_category' (Task B) *")
    pprint_label_distribution(label_dist_lc)

    # Get label distribution of the column 'label_vector' in training data
    label_dist_lv = get_label_distribution(train_df, 'label_vector')
    print("* Label distribution 'label_vector' (Task C) *")
    pprint_label_distribution(label_dist_lv)

    divider()

    # Get distribution of sentences containing emojis
    dist_e = sents_containing_emoji(train_df, 'text')
    print("* Distribution of sentences containing emojis - All *")
    pprint_label_distribution(dist_e)

    dist_e_sexist = sents_containing_emoji(train_df_sexist, 'text')
    print("* Distribution of sentences containing emojis - Sexist *")
    pprint_label_distribution(dist_e_sexist)

    dist_e_not_sexist = sents_containing_emoji(train_df_not_sexist, 'text')
    print("* Distribution of sentences containing emojis - Not sexist *")
    pprint_label_distribution(dist_e_not_sexist)

    divider()

    # Average length of texts in training data in words
    length_w_ls = get_avg_length(train_df, 'text')
    print("Average text length (in words) - All:", length_w_ls)

    length_w_ls_sexist = get_avg_length(train_df_sexist, 'text')
    print("Average text length (in words) - Sexist:", length_w_ls_sexist)

    length_w_ls_not_sexist = get_avg_length(train_df_not_sexist, 'text')
    print("Average text length (in words) - Not sexist:", length_w_ls_not_sexist, "\n")


if __name__ == "__main__":
    main()
