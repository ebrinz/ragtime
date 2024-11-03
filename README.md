# Development Exercise for Mistral-based RAG

This repository is an exercise in developing Retrieval-Augmented Generation (RAG) using the Mistral 7B LLM and Milvus database. The application is designed to run locally within Docker containers.

## Dataset
The dataset used for this project:
[Wikipedia Movie Plots - Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)

## Getting Started

To set up the environment and get started with the project, follow these steps:

1. Clone the repository and navigate to its directory.
2. Initialize and activate the environment:

    ```bash
    pipenv --python 3.11
    pipenv install
    pipenv run python -m ipykernel install --user --name="da_$(basename $(pwd))" --display-name="da_$(basename $(pwd))"
    ```

3. **Restart your code editor** to recognize the new environment.

4. **Open** the `main.ipynb` notebook and select the appropriate kernel and interpreter.

5. **Run the notebook** to start interacting with the RAG setup.

For additional notes and development progress, refer to `devjournal.ipynb`.

## Project Structure

- `main.ipynb` - The primary notebook for running RAG processes.
- `devjournal.ipynb` - The development work log and notes.
