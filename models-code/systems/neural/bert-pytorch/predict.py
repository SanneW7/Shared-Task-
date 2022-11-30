import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import argparse
import pickle
from train import LitModel
import torch

import sys
sys.path.append("../")

from utils import read_corpus
from bert_utils import BertDataset

"""## INFERENCE from checkpoint"""

class Inference_LitOffData(pl.LightningDataModule):
    def __init__(self, 
                 test_file: str = 'data/test.tsv',
                 batch_size = 4,
                 max_seq_len = 100,
                 modelname = 'distilbert-base-uncased',
                 encoderpath = 'encoder.pickle',
                ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.test_file = test_file
        self.get_encoder(encoderpath)
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.read_data()
        self.numerize_labels()
        self.setup()

    def read_test_data(self):
        # Read in the data
        X_test, Y_test = read_corpus(self.test_filename, ",", self.task_type)
        return X_test, Y_test

    def get_encoder(self, encoderpath):
        with open(encoderpath, "rb") as fp:
            self.encoder = pickle.load(fp)

    def numerize_labels(self):
        # Transform string labels to one-hot encodings
        self.Y_test_bin = self.encoder.transform(self.Y_test)

    def read_data(self):
        # Read in the data
        self.X_test, self.Y_test = read_corpus(self.test_file)

    def setup(self, stage = None):
        self.test_dataset= BertDataset(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                       texts = self.X_test, labels = self.Y_test_bin)

    def test_dataloader(self):
        dataloader=DataLoader(dataset=self.test_dataset,batch_size=self.batch_size)    
        return dataloader



def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_file", type=str, default='../../data/test.tsv', required= True,
                        help="If added, use trained model to predict on test set")

    parser.add_argument("--output_predfile", type=str, default='preds.txt', required= True,
                        help="File to store the predictions. Each prediction in a line")

    parser.add_argument("--debug_file", type=str, default='debug.csv',
                        help="Shows failed instances and all their predictions")

    parser.add_argument("--best_modelname", default="bert-files/bestmodel.ckpt", type=str,
                        help="Name of the trained model that will be saved after training")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")

    parser.add_argument("--device", default="gpu", type=str,
                        help="Type of device to use. gpu/cpu strict naming convention")

    args = parser.parse_args()
    return args


def main():
    '''Main function to test neural network given cmd line arguments'''
    args = create_arg_parser()
    testdm = Inference_LitOffData(test_file = args.test_file)
    model = LitModel.load_from_checkpoint(args.best_modelname,
                                          modelname = "bert-base-uncased")
    model.eval()
    device_to_train = args.device if torch.cuda.is_available() else "cpu"
    print("Device to use ", device_to_train)
    trainer = pl.Trainer(accelerator=device_to_train, devices=1)
    outs = trainer.predict(model, testdm.test_dataloader())
    print(outs[:4])

if __name__ == '__main__':
    main()

# testdm = Inference_LitOffData(test_file = args.test_file, modelname = args.langmodel_name)

# modelx = model.load_from_checkpoint(checkpoint_callback.best_model_path, modelname = args.langmodel_name, 
#                  learning_rate = args.learning_rate, 
#                  class_weights = class_weights,
#                  batch_size = args.batch_size)
# # 