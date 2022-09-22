import pandas as pd


def pprint_label_distribution(label_distribution):
    """Pretty-prints a label distribution"""
    tot_entries = label_distribution.sum()
    for label_info in label_distribution.iteritems():
        label = label_info[0]
        count = label_info[1]
        pct = '{:.2%}'.format(count / tot_entries)
        print('\"{0}\": {1} ({2})'.format(label, count, pct))


def get_label_distribution(df, label):
    """Return a Series containing counts of a label in a DataFrame column"""
    return df[label].value_counts()


def main():
    # Load training data
    train_path = "starting_kit/train_all_tasks.csv"
    train_df = pd.read_csv(train_path)

    # Get label distribution of the column 'label_sexist' in training data
    label_dist_a = get_label_distribution(train_df, 'label_sexist')
    pprint_label_distribution(label_dist_a)


if __name__ == "__main__":
    main()
