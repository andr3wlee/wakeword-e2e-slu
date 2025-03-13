import os
import shutil

from process_audio import turn_waves_to_mfcc
from train_model import train


def prepare_mfccs(path_to_snips):
    """
    Prepare MFCCs from WAV files.
    This function handles the conversion of audio files to MFCC features.
    """
    # Define paths
    wav_path = os.path.join(path_to_snips, "audio_files")
    mfcc_path = os.path.join(path_to_snips, "mfcc")

    # Ensure audio_files directory exists
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio files directory not found at {wav_path}")

    # Remove any existing MFCC directory and recreate it
    if os.path.exists(mfcc_path):
        print(f"Removing existing MFCC directory: {mfcc_path}")
        shutil.rmtree(mfcc_path)  # Use shutil for better cross-platform support

    # Create the mfcc directory
    os.makedirs(mfcc_path, exist_ok=True)

    # Create a temporary wav directory for MFCC conversion
    temp_wav_dir = os.path.join(path_to_snips, "wav")
    if os.path.exists(temp_wav_dir):
        shutil.rmtree(temp_wav_dir)

    # Copy audio files to the temporary wav directory
    os.makedirs(temp_wav_dir, exist_ok=True)
    for audio_file in os.listdir(wav_path):
        if audio_file.endswith(".wav"):
            src = os.path.join(wav_path, audio_file)
            dst = os.path.join(temp_wav_dir, audio_file)
            shutil.copy2(src, dst)

    # Run MFCC conversion
    print(f"Generating MFCCs in: {mfcc_path}")
    turn_waves_to_mfcc(path_to_snips, numcep=20)

    # Clean up the temporary directory
    if os.path.exists(temp_wav_dir):
        shutil.rmtree(temp_wav_dir)


if __name__ == '__main__':
    # Get the path to the root directory of the project
    root_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root_dir, "hey_snips_fl_amt")

    # Make sure the weights directory exists
    weights_dir = os.path.join(root_dir, "weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Prepare MFCCs and train the model
    prepare_mfccs(path)
    train()
