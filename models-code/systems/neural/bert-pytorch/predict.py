import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import argparse
import pickle
from train import LitModel
import torch

import sys
sys.path.append("../")

from utils import read_corpus, filter_none_class, extract_features
from bert_utils import BertDataset
from evaluate import get_encoder

"""## INFERENCE from checkpoint"""

class Inference_LitOffData(pl.LightningDataModule):
    def __init__(self, 
                 encoder,
                 test_file: str = 'data/test.tsv',
                 batch_size = 4,
                 max_seq_len = 100,
                 modelname = 'distilbert-base-uncased',
                 task_type = 'A',
                 perspective_filename = 'papi.json',
                 hurtlex_filename = 'hurtlex_features.json',
                 empath_filename = 'empath.json',
                ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.test_file = test_file
        self.encoder = encoder
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.task_type = task_type
        self.perspective_filename = perspective_filename
        self.hurtlex_filename = hurtlex_filename
        self.empath_filename = empath_filename
        self.read_data()
        self.read_features()
        self.numerize_labels()
        self.setup()

    def numerize_labels(self):
        # Transform string labels to one-hot encodings
        self.Y_test_bin = self.encoder.transform(self.Y_test)

    def read_data(self):
        # Read in the data
        self.test_ids, self.X_test, self.Y_test = read_corpus(self.test_file, ",",  self.task_type)
        if self.task_type != "A":
            self.test_ids, self.X_test, self.Y_test = filter_none_class(self.X_test, self.Y_test)

    def read_features(self):
        self.additional_test_features  = extract_features(self.test_ids, self.perspective_filename.replace(".json", "_test.json"),
                                               self.hurtlex_filename.replace(".json", "_test.json"),
                                               self.empath_filename.replace(".json", "_test.json"))

    def setup(self, stage = None):
        self.test_dataset= BertDataset(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                       texts = self.X_test, labels = self.Y_test_bin,
                                       additional_features = self.additional_test_features)

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

def main():
    '''Main function to test neural network given cmd line arguments'''
    args = create_arg_parser()
    encoder, basemodelname,\
    numlabels, task_type, model_extra_feat_len = get_encoder(f"{args.best_modelname}/details_{args.task_type}.pickle")
    print(task_type, args.task_type, basemodelname, numlabels)
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
                                         extra_feature_len=model_extra_feat_len)
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