import os
import sys
import torch
import soundfile as sf
import numpy as np
from slu_data_loader import read_config, get_SLU_datasets
from slu_models import Model


def perform_inference(audio_file_path, config_path="experiments/no_unfreezing.cfg"):
    """
    Perform inference with a trained SLU model on an audio file.

    Args:
        audio_file_path (str): Path to the audio file to process
        config_path (str): Path to the configuration file

    Returns:
        dict: Detected intents
    """
    # Check if files exist
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    print(f"Using config: {config_path}")
    config = read_config(config_path)

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get SLU datasets to initialize Sy_intent
    _, _, _ = get_SLU_datasets(config)

    # Initialize model
    model = Model(config)
    model.eval()

    # Load model state
    model_state_path = os.path.join(config.folder, "training", "model_state.pth")
    if not os.path.exists(model_state_path):
        raise FileNotFoundError(f"Model state file not found: {model_state_path}")

    print(f"Loading model state from: {model_state_path}")
    model.load_state_dict(torch.load(model_state_path, map_location=device))

    # Load and process audio
    print(f"Processing audio file: {audio_file_path}")
    signal, sample_rate = sf.read(audio_file_path)

    # Convert to tensor and add batch dimension
    signal = torch.tensor(signal, device=device).float().unsqueeze(0)

    # Perform inference
    intents = model.decode_intents(signal)

    print("Inference complete!")
    print(f"Detected intents: {intents}")

    return intents


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform inference with a trained SLU model")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--config", default="experiments/no_unfreezing.cfg",
                        help="Path to the configuration file (default: experiments/no_unfreezing.cfg)")

    args = parser.parse_args()

    perform_inference(args.audio_file, args.config)