import glob
import os

import numpy as np
from python_speech_features import mfcc
import progressbar
import scipy.io.wavfile as wav
import glob
import os

import numpy as np
from python_speech_features import mfcc
import progressbar
import scipy.io.wavfile as wav


def turn_waves_to_mfcc(path_to_waves, numcep=13):
    """
    Convert WAV files to MFCC features and save them as .npy files.

    Args:
        path_to_waves: Path to the directory containing the wav folder with WAV files
        numcep: Number of cepstral coefficients to compute
    """
    path_to_mfcc = make_sure_mfcc_path_exists(path_to_waves)
    wave_files_paths = glob.glob(os.path.join(path_to_waves, "wav", "*.wav"))

    if not wave_files_paths:
        print(f"No WAV files found in {os.path.join(path_to_waves, 'wav')}")
        return

    print(f"Found {len(wave_files_paths)} WAV files to process")

    for idx, wave_path in enumerate(progressbar.progressbar(wave_files_paths)):
        # Extract the filename without extension
        filename = os.path.basename(wave_path).replace(".wav", "")
        mfcc_path = os.path.join(path_to_mfcc, f"{filename}.mfcc")

        if not os.path.exists(mfcc_path + ".npy"):
            try:
                (rate, sig) = wav.read(wave_path)
                if len(sig) > 0:
                    mfcc_feat = mfcc(sig, rate, numcep=numcep)
                    np.save(mfcc_path, mfcc_feat)
                else:
                    print(f"Warning: Empty audio file: {wave_path}")
            except Exception as e:
                print(f"Error processing {wave_path}: {str(e)}")


def make_sure_mfcc_path_exists(path_to_audio):
    """
    Ensure the MFCC directory exists in the same parent directory as the audio directory.

    Args:
        path_to_audio: Path containing the wav directory

    Returns:
        Path to the MFCC directory
    """
    path_to_mfcc = os.path.join(path_to_audio, "mfcc")
    if not os.path.exists(path_to_mfcc):
        os.makedirs(path_to_mfcc, exist_ok=True)
        print(f"Created MFCC directory: {path_to_mfcc}")
    return path_to_mfcc

def turn_waves_to_mfcc(path_to_waves, numcep=13):
    """
    Convert WAV files to MFCC features and save them as .npy files.

    Args:
        path_to_waves: Path to the directory containing the wav folder with WAV files
        numcep: Number of cepstral coefficients to compute
    """
    path_to_mfcc = make_sure_mfcc_path_exists(path_to_waves)
    wave_files_paths = glob.glob(os.path.join(path_to_waves, "wav", "*.wav"))

    if not wave_files_paths:
        print(f"No WAV files found in {os.path.join(path_to_waves, 'wav')}")
        return

    print(f"Found {len(wave_files_paths)} WAV files to process")

    for idx, wave_path in enumerate(progressbar.progressbar(wave_files_paths)):
        # Extract the filename without extension
        filename = os.path.basename(wave_path).replace(".wav", "")
        mfcc_path = os.path.join(path_to_mfcc, f"{filename}.mfcc")

        if not os.path.exists(mfcc_path + ".npy"):
            try:
                (rate, sig) = wav.read(wave_path)
                if len(sig) > 0:
                    mfcc_feat = mfcc(sig, rate, numcep=numcep)
                    np.save(mfcc_path, mfcc_feat)
                else:
                    print(f"Warning: Empty audio file: {wave_path}")
            except Exception as e:
                print(f"Error processing {wave_path}: {str(e)}")


def make_sure_mfcc_path_exists(path_to_audio):
    """
    Ensure the MFCC directory exists in the same parent directory as the audio directory.

    Args:
        path_to_audio: Path containing the wav directory

    Returns:
        Path to the MFCC directory
    """
    path_to_mfcc = os.path.join(path_to_audio, "mfcc")
    if not os.path.exists(path_to_mfcc):
        os.makedirs(path_to_mfcc, exist_ok=True)
        print(f"Created MFCC directory: {path_to_mfcc}")
    return path_to_mfcc