## Development Exercise for Mistral based RAG

Purpose of this repo is perfome an exercise in RAG development using HyDE feature

Will run local Mistral Instruct 7b LLM and Milvus db in Docker

Set up vectorized embeddings and figure out a metric to compare RAG search vs HyDE search

Dataset used:
https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots

## Getting Starte

from repo ...
```
pipenv --python 3.11
pipenv install
pipenv run python -m ipykernel install --user --name="da_$(basename $(pwd))" --display-name="da_$(basename $(pwd))"
```

- Restart your editor

- Open main.ipynb notebook and select kernel and interpreter

- Run

- Refer to devjournal.ipynb for development work log

## Outcomes

TBD


