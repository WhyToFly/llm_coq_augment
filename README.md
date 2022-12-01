# Can Language Models Teach Themselves to Prove Better?
Project for CS 386L: Programming Languages. Based on ["Language Models Can Teach Themselves to Program Better" (Patrick Haluptzok, Matthew Bowers, Adam Tauman Kalai)](https://arxiv.org/abs/2207.14502)

## Setup (for Ubuntu 22.04.1 LTS)
- Install dependencies for [pycoq](https://github.com/IBM/pycoq)
- Use conda to install the "coq" and "train" environments from the yml files (for interacting with coq/testing the model and training the model respectively)
- gather theorem files from coq and for the test set (not provided as it contains class material and assignments)
- run `coq_parser.py` to extract theorems from the *.v files
- run `test_proof.py` to filter out theorems that don't compile
- fill in OpenAI API keys in `codex.py`
- run `training_scripts/splits.py` to create a train/validation/test split for the training data
- run `train.sh` to train the model; fill in the path for the checkpoint for the `model_name_or_path` parameter in the training script for subsequent training sessions and the `model_path` variable in `gpt_neo.py`

## Usage
- use `codex.py eval` to evaluate the test set using Codex
- use `gpt_neo.py eval` to evaluate the test set using GPT-Neo
- use `codex.py generate` to generate more theorems using Codex
- use `gpt_neo.py generate` to generate more theorems using GPT-Neo
