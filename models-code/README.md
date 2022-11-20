# Explain-preds
Code repo of the final LFD paper

## Setup

### Installing the requirements

```
conda create -n lfdenvg8 python=3.9
conda activate lfdenvg8
pip install -r requirements.txt
```

## Structure

* Discriminatory models are present inside "baselines/" folder

```
├── data
├── systems
│   ├── neural
│   │   ├── **bert/**
│   │   └── utils.py
├── LICENSE
├── README.md
├── requirements.txt
```
## Running baselines

1. Navigate to "systems/" folder for running them. The README.md inside it will guide on how to train/evaluate/predict.
