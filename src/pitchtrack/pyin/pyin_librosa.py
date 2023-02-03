import glob
import os
from pprint import pprint
from librosa import pyin as lpyin
from librosa import frames_to_time
import yaml
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import soundfile as sf
import pandas as pd
from librosa import load as lload
import numpy as np
import pandas as pd

def read_yaml_to_dict(path: str) -> dict:
    """
    Read YAML file into a dictionary

    Parameters
    ----------
    root : str
        root directory of the FBA project
    path_relative_to_root : str
        path of the YAML file, relative to `root`

    Returns
    -------
    dict
        Dictionary content of the YAML file
    """
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def pyin(
    inout,
    config_path="/media/fba/MIG-MusicPerformanceAnalysis-Code/src/pitchtrack/configs/new-default.yaml",
    verbose=True,
):
    wav_path, pyin_path = inout

    configs = read_yaml_to_dict(config_path)["pyin"]
    x, fs = lload(wav_path, sr=configs["sr"], res_type="polyphase")
    
    
    if verbose:
        print(f"Loaded {wav_path}...")

    f0, voice_flag, voice_prob = lpyin(x, **configs)
    t = frames_to_time(range(len(f0)), sr=fs, hop_length=configs["hop_length"], n_fft=configs["frame_length"])

    if verbose:
        print(f"Computed f0 for {wav_path}...")

    df = pd.DataFrame({"time": t, "f0": f0, "voice_flag": voice_flag, "voice_prob": voice_prob})

    df.to_csv(pyin_path, index=False)

    if verbose:
        print(f"Saved {pyin_path}...")
    # except:
        # print(f"Error in {wav_path}")

def process(
    audio_root="/media/fba/MIG-FBA-Audio/cleaned/audio/bystudent",
    pyin_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack2",
    use_multiprocessing=False,
):

    wav_paths = sorted(glob.glob(audio_root + "/**/*.wav", recursive=True))
    pyin_paths = [
        wav_path.replace(audio_root, pyin_root).replace(".wav", ".csv")
        for wav_path in wav_paths
    ]

    for w in pyin_paths:
        folder = os.path.dirname(w)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    inout = list(zip(wav_paths, pyin_paths))
    if use_multiprocessing:
        process_map(pyin, inout, max_workers=1, chunksize=1, total=len(wav_paths))
    else:
        for i in tqdm(inout):
            pyin(i)

def checkfs(wav_path):
    _, fs = sf.read(wav_path, stop=1)
    return {'file': wav_path, 'fs': fs}

def checkfs_all(audio_root="/media/fba/MIG-FBA-Audio/cleaned/audio/bystudent"):
    wav_paths = sorted(glob.glob(audio_root + "/**/*.wav", recursive=True))
    fs = process_map(checkfs, wav_paths, max_workers=64, chunksize=128, total=len(wav_paths))

    df = pd.DataFrame(fs)
    df.to_csv('fs.csv', index=False)

    pprint(df['fs'].value_counts())

if __name__ == "__main__":
    import fire
    fire.Fire()