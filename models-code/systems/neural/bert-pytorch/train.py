# -*- coding: utf-8 -*-
"""SharedTask bert-pytorch lightining.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15akolCxsGCIcRQVh77b4gbrNRkFJ4S6S
"""

from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torchmetrics
import argparse
import pickle
import numpy as np

import sys
sys.path.append("../")

from utils import compute_class_weights, read_corpus,\
                  filter_none_class, numerize_labels_pytorch,\
                  read_json, extract_features

from bert_utils import BertDataset


class LitOffData(pl.LightningDataModule):
    def __init__(self, train_file: str = 'data/train.tsv',
                 dev_file: str = 'data/dev.tsv',
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
        self.train_file, self.dev_file = train_file, dev_file
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.task_type = task_type
        self.perspective_filename = perspective_filename
        self.hurtlex_filename = hurtlex_filename
        self.empath_filename = empath_filename
        self.read_data()
        self.read_features()
        self.encoder, self.Y_train_bin, \
        self.Y_dev_bin = numerize_labels_pytorch(self.Y_train, self.Y_dev)

    def read_data(self):
        # Read in the data
        self.train_ids, self.X_train, self.Y_train = read_corpus(self.train_file, ",",  self.task_type)
        self.dev_ids, self.X_dev, self.Y_dev = read_corpus(self.dev_file, ",", self.task_type)
        if self.task_type != "A":
            self.train_ids, self.X_train, self.Y_train = filter_none_class(self.train_ids, self.X_train, self.Y_train)
            self.dev_ids, self.X_dev, self.Y_dev = filter_none_class(self.dev_ids, self.X_dev, self.Y_dev)

    def read_features(self):
        self.additional_train_features  = extract_features(self.train_ids, self.perspective_filename.replace(".json", "_train.json"),
                                               self.hurtlex_filename.replace(".json", "_train.json"),
                                               self.empath_filename.replace(".json", "_train.json"))
        
        # import pdb;
        # pdb.set_trace();
        self.additional_dev_features  = extract_features(self.dev_ids, self.perspective_filename.replace(".json", "_dev.json"),
                                               self.hurtlex_filename.replace(".json", "_dev.json"),
                                               self.empath_filename.replace(".json", "_dev.json"))
        
        self.extra_feat_len = self.additional_train_features.shape[1]

    def setup(self, stage = None):
        self.train_dataset= BertDataset(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                        texts = self.X_train, labels = self.Y_train_bin,
                                        additional_features=self.additional_train_features)
        self.val_dataset= BertDataset(tokenizer = self.tokenizer, max_length=self.max_seq_len,
                                      texts = self.X_dev, labels = self.Y_dev_bin,
                                      additional_features=self.additional_dev_features)

    def train_dataloader(self):
        dataloader=DataLoader(dataset=self.train_dataset,batch_size=self.batch_size)
        return dataloader

    def val_dataloader(self):
        dataloader=DataLoader(dataset=self.val_dataset,batch_size=self.batch_size)
        return dataloader

    def get_weights(self):
        self.class_weightscores = compute_class_weights(self.encoder, self.Y_train)
        print(self.class_weightscores)
        return torch.tensor(self.class_weightscores)

# define the LightningModule
class LitModel(pl.LightningModule):
    def __init__(self, modelname = "distilbert-base-uncased", num_labels = 2, dropout = 0.2,
                learning_rate = 1e-5, class_weights = [1, 1], batch_size = 4,
                extra_feature_len = 0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(modelname)
        self.linear = nn.Linear(768 + extra_feature_len, num_labels)
        print("Lit MODEL is", extra_feature_len,num_labels)
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.loss_fn =  nn.CrossEntropyLoss(weight = torch.tensor(self.class_weights, dtype=torch.float))
        self.accuracy = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1Score(num_classes=num_labels, average='macro')
        self.batch_size = batch_size

    def forward(self, **data):
        features = data["additional_features"]
        del data["target"]
        del data["additional_features"]
        out = self.bert( **data, return_dict=True)
        pooled_output = out.pooler_output
        dropout_output = nn.functional.dropout(pooled_output)
        # logits = self.linear(dropout_output)
        logits = self.linear(torch.cat((dropout_output, features), 1))
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        logits = self(**batch)
        loss = self.loss_fn(logits, batch["target"])

        # validation metrics
        preds = torch.nn.functional.softmax(logits, dim = -1)
        preds = torch.argmax(preds, -1)
        acc = self.accuracy(preds, batch["target"])
        f1 = self.f1(preds, batch["target"])

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.loss_fn(logits, batch["target"])

        # validation metrics
        preds = torch.nn.functional.softmax(logits, dim = -1)
        preds = torch.argmax(preds, -1)
        acc = self.accuracy(preds, batch["target"])
        f1 = self.f1(preds, batch["target"])

        self.log('val_loss', loss,on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', acc,on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_f1', f1,on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "gts": batch["target"]}

    def test_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.loss_fn(logits, batch["target"])

        # validation metrics
        preds = torch.nn.functional.softmax(logits, dim = -1)
        preds = torch.argmax(preds, -1)
        acc = self.accuracy(preds, batch["target"])
        f1 = self.f1(preds, batch["target"])

        self.log('test_loss', loss,on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_acc', acc,on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_f1', f1,on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "gts": batch["target"]}

    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(**batch)
        preds = torch.nn.functional.softmax(logits, dim = -1)
        preds = torch.argmax(preds, -1)
        return preds

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        losses = torch.stack([ item["loss"]  for item in outs])

        preds = []
        gts = []
        
        for item in outs:
            if item["preds"].shape[0]!= self.batch_size: continue
            preds.append(item["preds"])
            gts.append(item["gts"])
        
#         pdb.set_trace();

        preds = torch.stack(preds).view(-1)     
        gts = torch.stack(gts).view(-1)

        loss = losses.mean()
        acc = self.accuracy(preds, gts)
        f1 = self.f1(preds, gts)
        
        self.log("val_loss_epoch", loss)
        self.log("val_epoch_acc", acc)
        self.log("val_epoch_f1", f1)
        print("val_loss_epoch", loss)
        print("val_acc_epoch", acc)
        print("val_F1_epoch", f1)

    def test_epoch_end(self,outs):
        # outs is a list of whatever you returned in `validation_step`
        losses = torch.stack([ item["loss"]  for item in outs])
        
        preds = []
        gts = []
        
        for item in outs:
            if item["preds"].shape[0]!= self.batch_size: continue
            preds.append(item["preds"])
            gts.append(item["gts"])
        
        preds = torch.stack(preds).view(-1)     
        gts = torch.stack(gts).view(-1)

        loss = losses.mean()
        acc = self.accuracy(preds, gts)
        f1 = self.f1(preds, gts)

        self.log("test_loss_epoch", loss)
        self.log("test_epoch_acc", acc)
        self.log("test_epoch_f1", f1)
        print("test_loss_epoch", loss)
        print("test_acc_epoch", acc)
        print("test_F1_epoch", f1)
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--train_file", default='../../../../data/train.csv', type=str,
                        help="Input file to learn from (default train.txt)")
    
    parser.add_argument("-d", "--dev_file", type=str, default='../../../../data/dev.csv',
                        help="Separate dev set to read in (default dev.txt)")
    
    parser.add_argument("-t", "--test_file", type=str, default='../../../../data/test.csv',
                        help="If added, use trained model to predict on test set")
    
    parser.add_argument("-e", "--embeddings", default='glove_reviews.json', type=str,
                        help="Embedding file we are using (default glove_reviews.json)")
    
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Learning rate for the optimizer")

    parser.add_argument("--task_type", type=str, default="A",
                        help="A or B")

    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training")

    parser.add_argument("--num_epochs", default=2, type=int,
                        help="Number of epochs for training")

    parser.add_argument("--max_seq_len", default=150, type=int,
                        help="Maximum length of input sequence after BPE")

    parser.add_argument("--langmodel_name", default="bert-base-uncased", type=str,
                        help="Name of the base pretrained language model")

    parser.add_argument("--papi_name", default="", type=str,
                        help="Name of Perspective file name without train/dev/test names")

    parser.add_argument("--hurtlex_name", default="", type=str,
                        help="Name of Hurtlex file name without train/dev/test names")

    parser.add_argument("--empath_name", default="", type=str,
                        help="Name of Empath file name without train/dev/test names")

    parser.add_argument("--ckpt_folder", default="./bert-files/", type=str,
                        help="Name of the checkpoint folder for saving the model")


    args = parser.parse_args()
    return args

def main():
    '''Main function to test neural network given cmd line arguments'''
    args = create_arg_parser()
    print(args)

    ckpt_folder = args.ckpt_folder
    Path(ckpt_folder).mkdir(parents=True, exist_ok=True)

    dm = LitOffData(train_file =  args.train_file,
                    dev_file =  args.dev_file,
                    batch_size = args.batch_size,
                    max_seq_len = args.max_seq_len,
                    modelname = args.langmodel_name,
                    task_type = args.task_type,
                    perspective_filename=args.papi_name,
                    hurtlex_filename=args.hurtlex_name,
                    empath_filename=args.empath_name)

    class_weights = dm.get_weights()

    numlabels = len(set(dm.Y_train))
    model = LitModel(modelname = args.langmodel_name,
                    learning_rate = args.learning_rate,
                    num_labels= numlabels,
                    class_weights = class_weights,
                    batch_size = args.batch_size,
                    extra_feature_len=dm.extra_feat_len)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience = 3)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder,
                                          monitor="val_loss",
                                          mode="min",
                                          filename=f"bestmodel_{args.task_type}")

    trainer = pl.Trainer(accelerator='cpu', devices=1, 
                        max_epochs = args.num_epochs, fast_dev_run=False,
                        callbacks=[ early_stopping, checkpoint_callback ],
                        limit_train_batches=10, limit_val_batches=5)

    with open(f"{ckpt_folder}/details_{args.task_type}.pickle", "wb") as fh:
        pickle.dump([dm.encoder, args.langmodel_name,
                     numlabels, args.task_type, dm.extra_feat_len], fh)
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
