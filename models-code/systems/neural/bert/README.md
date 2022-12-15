
# Model BERT

Trains a BERT model using train & dev set.

## Running the model

1. To train the model

```
python train.py --train_file ../../../../data/train.csv --dev_file ../../../../data/dev.csv --learning_rate 5e-5 --max_seq_len 200 --num_epochs 1 --batch_size 4 --langmodel_name GroNLP/hateBERT --task_type A --empath_name ../../../../data/empath.json --hurtlex_name ../../../../data/hurtlex_features.json --papi_name ../../../../data/papi_features.json --output_modelname models/phe_bz4_ep1_5e5_A_hatebert
```


Evaluate script will automatically pick the task type from output modelname

2. To evaluate the model

```
python evaluate.py --test_file ../../../../data/test.csv --best_modelname models/phe_bz4_ep1_5e5_A_hatebert
```

3. To predict the model

```
python predict.py --test_file ../../../../data/dev_task_a_entries.csv --best_modelname models/phe_bz4_ep1_5e5_A_hatebert --output_predfile phe_bz4_ep1_5e5_A_hatebert.txt
```

If you want to try any other bert variant, please type

python train.py --langmodel_name distilbert-base-uncased

Or for other models, try these:

microsoft/deberta-v3-base
xlnet-base-cased
roberta-base
bert-base-uncased
distilbert-base-uncased
albert-base-v2

