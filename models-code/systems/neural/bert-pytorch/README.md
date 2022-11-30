
Usage
```
python train.py --task_type A --hurtlex_name ../../../../data/hurtlex_features.json --num_epochs 1 --ckpt_folder tst1

python predict.py --test_file ../../../../data/test.csv --output_predfile preds.txt --task_type A --best_modelname tst1/ --hurtlex_name ../../../../data/hurtlex_features.json
```