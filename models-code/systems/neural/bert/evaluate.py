import argparse

import sys
sys.path.append("../")

from utils import load_picklefile, test_set_predict
from bert_utils import load_model, read_testdata_andvectorize

def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_file", type=str, default='../../../data/test.tsv', required= True,
                        help="If added, use trained model to predict on test set")

    parser.add_argument("--best_modelname", default="models/bert-outputs", type=str,
                        help="Name of the trained model that will be saved after training")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")

    parser.add_argument("--show_cm", default=True, type=bool,
                        help="Show confusion matrix")

    args = parser.parse_args()
    return args

def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    print(args)
    encoder = load_picklefile(f"{args.best_modelname}.pickle")
    base_lm, max_seq_len, task_type = load_picklefile(f"{args.best_modelname}.details")
    best_model, tokenizer = load_model(base_lm, num_labels= len(encoder.classes_))
    best_model.load_weights(args.best_modelname)
    test_ids, X_test, Y_test, tokens_test, Y_test_bin = read_testdata_andvectorize(args.test_file, max_seq_len, tokenizer, encoder, task_type)
    Y_pred, Y_test = test_set_predict(best_model, tokens_test, Y_test_bin,
                    "test", encoder, showplot=args.show_cm, task_type=task_type)

if __name__ == '__main__':
    main()
