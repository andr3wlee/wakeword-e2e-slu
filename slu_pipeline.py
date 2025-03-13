import sys
import os
import torch
import numpy as np
import soundfile as sf
from slu_models import PretrainedModel, Model
from slu_data_loader import get_ASR_datasets, get_SLU_datasets, read_config
from slu_training import Trainer
import argparse

# Add the correct path to `key_words_spotting_master` so we can import wake word model
wake_word_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../key_words_spotting_master"))
if wake_word_path not in sys.path:
    sys.path.append(wake_word_path)

from keyword_spotter import KeyWordSpotter  # Import wake word model



# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', help='run ASR pre-training')
parser.add_argument('--train', action='store_true', help='run SLU training')
parser.add_argument('--infer', action='store_true', help='run inference on test audio')
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters, etc.')
parser.add_argument('--audio_path', type=str, help='path to test audio file for inference')

if __name__ == '__main__':
    args = parser.parse_args()
    pretrain = args.pretrain
    train = args.train
    infer = args.infer
    restart = args.restart
    config_path = args.config_path
    audio_path = args.audio_path

    print(f"Using config file: {config_path}")

    # Read config file
    config = read_config(config_path)
    torch.manual_seed(config.seed);
    np.random.seed(config.seed)

    if pretrain:
        train_dataset, valid_dataset, test_dataset = get_ASR_datasets(config)
        pretrained_model = PretrainedModel(config=config)
        trainer = Trainer(model=pretrained_model, config=config)
        if restart: trainer.load_checkpoint()
        for epoch in range(config.pretraining_num_epochs):
            print(f"========= Epoch {epoch + 1} of {config.pretraining_num_epochs} =========")
            trainer.train(train_dataset)
            trainer.test(valid_dataset)
            trainer.save_checkpoint()

    if train:
        train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config)
        model = Model(config=config)
        trainer = Trainer(model=model, config=config)
        if restart: trainer.load_checkpoint()
        for epoch in range(config.training_num_epochs):
            print(f"========= Epoch {epoch + 1} of {config.training_num_epochs} =========")
            trainer.train(train_dataset)
            trainer.test(valid_dataset)
            trainer.save_checkpoint()

    if infer:
        # ✅ Load Wake Word Model
        wake_word_model = KeyWordSpotter(20)
        wake_word_model.load_state_dict(
            torch.load("../key_words_spotting_master/weights/latest_keyword_model.pth", map_location="cpu"))
        wake_word_model.eval()

        # ✅ Load SLU Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(config).to(device).eval()
        model.load_state_dict(torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))

        # ✅ Process Audio for Wake Word
        signal, _ = sf.read(audio_path)
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        # ✅ Check for Wake Word Detection
        wake_word_detected = wake_word_model(signal_tensor).item() > 0.5  # Threshold-based detection

        if not wake_word_detected:
            print("Wake word NOT detected. SLU is not triggered.")
        else:
            print("Wake word detected! Running SLU model...")
            intents, entities = model.decode_intents(signal_tensor)
            print(f"Predicted Intent: {intents}, Entities: {entities}")

