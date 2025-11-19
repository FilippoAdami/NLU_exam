# NLU Exam Solution

## Overview
This repository contains the final solution for the Language Modeling assignment (Part 1.A and 1.B) and the Natural Language Understanding assignement (Part 2.A and 2.B).

## Directories Structure
* `main.py`: Orchestrator script. Configures hyperparameters, initializes models, and runs the training loop.
* `model.py`: Contains the PyTorch model definitions (`LM_RNN`, `LM_LSTM`, `LM_LSTM_VDO`, etc.).
* `utils.py`: Contains dataset loading (`PennTreeBank` class) and preprocessing (`Lang` class, `collate_fn`).
* `functions.py`: Contains training loops, evaluation loops, and weight initialization.
* `dataset/`: Folder containing the PTB text files.
* `bin/`: Folder where the best trained model for each step is saved.

## Additional Data
The files LM_hyperparameters_tuning_complete contains the walkthrough of the whole LM and NLU parts with all the hyperparameters that were tried and the reasoning behing my choices. Some hyperparameters combinations were excluded from the report to keep the report more clear and concise. The file has been added just as testimony of the work that has been done.

## Generative AI usage
Generative AI (Gemini) was used to help with code refactoring and comments.
