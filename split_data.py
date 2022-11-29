import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_data(df, seed):
    """Create train, dev, test sets with representative proportions

    9800 training examples

    2100 development examples

    2100 test examples
    """

    X = list(zip(df["rewire_id"],df["text"]))
    y_taskA = df["label_sexist"]
    y_taskB = df["label_category"]
    y_taskC = df["label_vector"]

    y = list(zip(y_taskA, y_taskB, y_taskC))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    # Split test set again for dev/test split
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def return_labelwise_data(y):
    y_A = [taskA for taskA, _, _ in y]
    y_B = [taskB for _, taskB, _ in y]
    y_C = [taskC for _, _, taskC in y]
    return y_A, y_B, y_C

def return_ids_text(X):
    ids = [id for id, _ in X]
    txts = [txt for _, txt in X]
    return ids, txts

def createfolder_ifnot(foldername):
    Path(foldername).mkdir(parents=True, exist_ok=True)

def write_tofiles(X_train, X_dev, X_test,\
                  y_train, y_dev, y_test,\
                  foldername):
    y_train_A, y_train_B, y_train_C = return_labelwise_data(y_train)
    y_dev_A, y_dev_B, y_dev_C = return_labelwise_data(y_dev)
    y_test_A, y_test_B, y_test_C = return_labelwise_data(y_test)

    ids_train, X_train = return_ids_text(X_train)
    ids_dev, X_dev = return_ids_text(X_dev)
    ids_test, X_test = return_ids_text(X_test)

    train = pd.DataFrame(list(zip(ids_train, X_train, y_train_A, y_train_B, y_train_C)),
                         columns = ["rewire_id", "text", "label_sexist", "label_category", "label_vector"])
    dev = pd.DataFrame(list(zip(ids_dev, X_dev, y_dev_A, y_dev_B, y_dev_C)),
                         columns = ["rewire_id", "text", "label_sexist", "label_category", "label_vector"])
    test = pd.DataFrame(list(zip(ids_test, X_test, y_test_A, y_test_B, y_test_C)),
                         columns = ["rewire_id", "text", "label_sexist", "label_category", "label_vector"])
    createfolder_ifnot(foldername)
    train.to_csv(f"{foldername}/train.csv", index=False)
    dev.to_csv(f"{foldername}/dev.csv", index=False)
    test.to_csv(f"{foldername}/test.csv", index=False)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='./starting_kit/train_all_tasks.csv', type=str,
                        help="File containing the reviews, which are split up later")
    parser.add_argument("--output_foldername", default='./data/', type=str,
                        help="Folder where train/dev/test splits are saved")
    parser.add_argument("--seed", type=int, default=120,
                        help="Seed for random state for experiments")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = create_arg_parser()
    np.random.seed(args.seed)
    print(args)
    df = pd.read_csv(args.input_file)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df, args.seed)
    write_tofiles(X_train, X_dev, X_test,\
                    y_train, y_dev, y_test,\
                    args.output_foldername)