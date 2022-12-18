import argparse

import sys
sys.path.append("../")

from utils import load_picklefile, get_preds, write_preds, add_features_to_sents
from bert_utils import load_model, read_testdata_andvectorize

def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--test_file", type=str, default='../../../data/test.tsv', required= True,
                        help="If added, use trained model to predict on test set")

    parser.add_argument("--best_modelname", default="models/bert-outputs", type=str,
                        help="Name of the trained model that will be saved after training")

    parser.add_argument("--output_predfile", type=str, default='preds.csv', required= True,
                        help="File to store the predictions. Each prediction in a line")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")

    parser.add_argument("--show_cm", default=True, type=bool,
                        help="Show confusion matrix")
   
    parser.add_argument("--task_type", type=str, default="A",
                        help="A or B")

    args = parser.parse_args()
    return args

def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    print(args)
    base_lm, max_seq_len, task_type, feature_paths, threshold_values = load_picklefile(f"{args.best_modelname}_task_{args.task_type}.details")
    assert task_type==args.task_type, "Make sure correct model files are passed\n Check task type"
    encoder = load_picklefile(f"{args.best_modelname}_task_{task_type}.pickle")
    best_model, tokenizer = load_model(base_lm, num_labels= len(encoder.classes_))
    best_model.load_weights(f"{args.best_modelname}_task_{task_type}")
    test_ids, X_test, Y_test, tokens_test = read_testdata_andvectorize(args.test_file, max_seq_len, tokenizer, encoder, task_type)
    if feature_paths["empath"]:
        test_ids, X_test = add_features_to_sents(test_ids, X_test, f"dev_task_{task_type.lower()}_entries", feature_paths["empath"], "empath", threshold_values)
    if feature_paths["hurtlex"]:
        test_ids, X_test = add_features_to_sents(test_ids, X_test, f"dev_task_{task_type.lower()}_entries", feature_paths["hurtlex"], "hurtlex", threshold_values)
    if feature_paths["papi"]:
        test_ids, X_test = add_features_to_sents(test_ids, X_test, f"dev_task_{task_type.lower()}_entries", feature_paths["papi"], "papi", threshold_values)
    Y_pred = get_preds(model = best_model, X_test =  tokens_test, task_type=task_type,encoder= encoder)
    write_preds(test_ids, Y_pred, args.output_predfile)
    print("Predictions done!!")

if __name__ == '__main__':
    main()
