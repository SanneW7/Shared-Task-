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

Negative stereotypes and ethnic slurs (PS)

Professions and occupations (PA)

Physical disabilities and diversity (DDF)

Cognitive disabilities and diversity (DDP)

Female genitalia (ASF)

Words related to prostitution (PR)

Words related to homosexuality (OM)

With potential negative connotations (QAS)

Derogatory words (CDS)

Male genitalia (ASM)

### PerspectiveAPI
PerspectiveAPI features are represented by a vector consisting of 9 values. The categories corresponding to each value are in alphabetical order; "FLIRTATION", "IDENTITY_ATTACK", "INSULT", "OBSCENE", "PROFANITY", "SEVERE_TOXICITY", "SEXUALLY_EXPLICIT", "THREAT" and "TOXICITY".

### Empath
The order of the used categories is as follows: "sexism", "violence", "money", "valuable" "domestic work", "hate", "aggression", "anticipation", "crime", "weakness",
                 "horror", "swearing terms", "kill", "sexual", "cooking",
                 "exasperation", "body", "ridicule", "disgust", "anger", "rage"
