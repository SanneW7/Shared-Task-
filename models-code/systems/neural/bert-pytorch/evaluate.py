import pytorch_lightning as pl
import argparse
import pickle
from train import LitModel
import torch

import sys
sys.path.append("../")

from predict import Inference_LitOffData
"""## INFERENCE from checkpoint"""


def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_file", type=str, default='../../data/test.tsv', required= True,
                        help="If added, use trained model to predict on test set")

    parser.add_argument("--debug_file", type=str, default='debug.csv',
                        help="Shows failed instances and all their predictions")

    parser.add_argument("--best_modelname", default="bert-files/", type=str,
                        help="Name of the trained model that will be saved after training")

    parser.add_argument("--papi_name", default="", type=str,
                        help="Name of Perspective file name without train/dev/test names")

    parser.add_argument("--hurtlex_name", default="", type=str,
                        help="Name of Hurtlex file name without train/dev/test names")

    parser.add_argument("--empath_name", default="", type=str,
                        help="Name of Empath file name without train/dev/test names")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")

    parser.add_argument("--task_type", type=str, default="A",
                        help="A or B")

    parser.add_argument("--device", default="gpu", type=str,
                        help="Type of device to use. gpu/cpu strict naming convention")

    args = parser.parse_args()
    return args


def get_encoder(detailsfile):
    with open(detailsfile, "rb") as fp:
        encoder, modelname, numlabels, task_type, extra_feat_len, dropout = pickle.load(fp)
    return encoder, modelname, numlabels, task_type, extra_feat_len, dropout

def main():
    '''Main function to test neural network given cmd line arguments'''
    args = create_arg_parser()
    encoder, basemodelname,\
    numlabels, task_type, model_extra_feat_len, dropout = get_encoder(f"{args.best_modelname}/details_{args.task_type}.pickle")
    print(task_type, args.task_type, basemodelname, numlabels, model_extra_feat_len)
    assert task_type==args.task_type, "Make sure correct model files are passed"
    testdm = Inference_LitOffData(test_file = args.test_file,
                                  encoder = encoder,
                                  task_type = task_type,
                                  perspective_filename=args.papi_name,
                                  hurtlex_filename=args.hurtlex_name,
                                  empath_filename=args.empath_name)
    model = LitModel.load_from_checkpoint(f"{args.best_modelname}/bestmodel_{task_type}.ckpt", 
                                         modelname = basemodelname, num_labels = numlabels,
                                         class_weights = [1]*numlabels,
                                         extra_feature_len=model_extra_feat_len,
                                         dropout = )
    model.eval()
    device_to_train = args.device if torch.cuda.is_available() else "cpu"
    print("Device to use ", device_to_train)
    trainer = pl.Trainer(accelerator=device_to_train, devices=1)
    trainer.test(model, testdm.test_dataloader())

if __name__ == '__main__':
    main()