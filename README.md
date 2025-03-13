# wakeword-e2e-slu

This repository contains the code for an end-to-end Spoken Language Understanding (SLU) and Natural Language Understanding (NLU) pipeline using wake word detection. It is built as part of the CS 224n project and utilizes datasets like the **Hey Snips** wake word dataset and the **Fluent AI** dataset for training.

## Prerequisites

1. **Datasets**:
   - You will need to download the following datasets:
     - [Hey Snips Wake Word Dataset](https://github.com/sonos/keyword-spotting-research-datasets)
     - [Fluent AI Speech Commands Dataset](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/)
   
2. **System Requirements**:
   - Google Colab with T4 or L4 GPU is recommended for optimal performance.
   
## Setup Instructions

1. **Acquire Datasets**:
   - Download the **Hey Snips** and **Fluent AI** datasets from the links provided above.
   - Extract the zip files into appropriate directories.

2. **Directory Structure**:
   - Place the **Hey Snips** dataset in the specified directory for training the wake word model.
   - After extracting the **Fluent AI** dataset, create a folder called `data/` in the root of the project, and place the unzipped dataset inside this folder.

## Training the Wake Word Model

1. Once the **Hey Snips** dataset is set up, you can train the wake word model by running the following command:
   ```bash
   python run_training.py
   ```

2. This will initiate the training process for the Hey Snips wake word detection model. Make sure the datasets are in the correct location before running the command.

## Training the SLU Model

1. After training the wake word model, extract the dataset for **Fluent AI**, which will create a new `data/` folder containing the relevant files.

2. To train the Spoken Language Understanding (SLU) model, run the following command:
   ```bash
   python slu_pipeline.py --train --config_path=experiments/no_unfreezing.cfg
   ```

## Running Inference

After the training process is complete, you can run inference on a sample audio file:

1. Use the following command to run the SLU inference pipeline:
   ```bash
   python slu_inference.py tests_audio.wav --config experiments/no_unfreezing.cfg
   ```

2. This will process the provided `tests_audio.wav` and output the results.

## Notes

- The **Hey Snips** dataset is used for wake word detection, and the **Fluent AI** dataset is used for Spoken Language Understanding (SLU).
- The project integrates wake word detection and SLU into an end-to-end pipeline, making it suitable for voice-based applications.



