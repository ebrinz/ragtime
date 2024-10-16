## Dev Environment for Mistral based RAG


### instructs

from repo ...
```
pipenv --python 3.11
pipenv install
pipenv run python -m ipykernel install --user --name="da_$(basename $(pwd))" --display-name="da_$(basename $(pwd))"
```

Restart your editor

open notbook and select kernel and interpreter


