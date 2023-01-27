import glob
import os
from librosa import pyin as lpyin
from librosa import frames_to_time
import yaml
from tqdm.contrib.concurrent import process_map
import soundfile as sf
import pandas as pd
from librosa import load as lload

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
    config_path="/media/fba/MIG-MusicPerformanceAnalysis-Code/src/pitchtrack/configs/new.yaml",
):

    wav_path, pyin_path = inout

    configs = read_yaml_to_dict(config_path)["pyin"]
    x, fs = lload(wav_path, sr=configs["sr"])

    f0, voice_flag, voice_prob = lpyin(x, **configs)
    t = frames_to_time(range(len(f0)), sr=fs, hop_length=configs["hop_length"], n_fft=configs["frame_length"])

    df = pd.DataFrame({"time": t, "f0": f0, "voice_flag": voice_flag, "voice_prob": voice_prob})

    df.to_csv(pyin_path, index=False)


def process(
    audio_root="/media/fba/MIG-FBA-Audio/cleaned/audio/bystudent",
    pyin_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack2",
):

    wav_paths = glob.glob(audio_root + "/**/*.wav", recursive=True)
    pyin_paths = [
        wav_path.replace(audio_root, pyin_root).replace(".wav", ".csv")
        for wav_path in wav_paths
    ]

    for w in wav_paths:
        folder = os.path.dirname(w)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    inout = list(zip(wav_paths, pyin_paths))

    process_map(pyin, inout, max_workers=8, chunksize=1, total=len(wav_paths))

if __name__ == "__main__":
    import fire
    fire.Fire(process)