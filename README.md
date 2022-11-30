## Requirements

Install the requirements by running the following command,

```
pip install -r requirements.txt
```

## Usage

```python

python code.py --input_file  ./starting_kit/train_all_tasks.csv --task_type A --vectorizer tf

```

For best results,

```python
python code.py --input_file  ./starting_kit/train_all_tasks.csv --task_type A --vectorizer tf

python code.py --input_file  ./starting_kit/train_all_tasks.csv --task_type B --vectorizer tfidf

```

## Data

### Hurtlex
The order of the used categories is as follows: "ps", "pa", "ddf", "ddp", "asf", "pr", "om", "qas", "cds", "asm"