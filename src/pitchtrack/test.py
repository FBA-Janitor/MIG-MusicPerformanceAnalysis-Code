import os
from pyin import pyin_librosa

import soundfile as sf

import sys

sys.path.append("/media/fba/MIG-MusicPerformanceAnalysis-Code/src/data_parse/scripts")

from utils import default_configs_path

import pandas as pd
import numpy as np

from tqdm import tqdm

import librosa

from scipy.signal import correlate, correlation_lags



def check_consistency(audio_path, old_pitchtrack_path):

    # x, fs = sf.read(audio_path)
    x, fs = librosa.load(
        audio_path,
        sr=None,
        mono=False
    )
    
    
    if len(x.shape) > 1:
        # Ashis used only the first channel
        x = x[0, :]
    # print(fs)
    
    # x = np.concatenate([np.zeros((512,)), x])
    
    f0_old = pd.read_csv(old_pitchtrack_path)["MIDI"].to_numpy()
    f0_old[f0_old == 0.0] = np.nan

    f0_librosa, vflag, vprob = pyin_librosa.pyin(x, fs=fs, center=True)

    f0_librosa = np.round(f0_librosa, decimals=2)

    min_shape = min(f0_librosa.shape[0], f0_old.shape[0])
    
    pad_adjust = 2
    
    f0_librosa = f0_librosa[:min_shape-1]
    f0_old = f0_old[pad_adjust:min_shape+pad_adjust-1]

    midi_librosa = librosa.hz_to_midi(f0_librosa)
    midi_old = librosa.hz_to_midi(f0_old)
    
    midi_librosa[np.isnan(midi_librosa)] = 0
    midi_old[np.isnan(midi_old)] = 0
    
    cidx = np.argmax(correlate(midi_librosa, midi_old))
    lags = correlation_lags(min_shape, min_shape)
    
    print(lags[cidx])

    out = np.mean(np.isclose(f0_librosa, f0_old, atol=0.5))

    print(np.stack([f0_librosa, f0_old]))

    return out


def process(year, band):

    pitchtrack_root = "/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack/bystudent"
    audio_root = "/media/fba/MIG-FBA-Data-Cleaning/cleaned/audio/bystudent"

    pt_folder = os.path.join(pitchtrack_root, str(year), band)
    audio_folder = os.path.join(audio_root, str(year), band)

    pt_students = os.listdir(pt_folder)
    audio_students = os.listdir(audio_folder)

    students = sorted(list(set(pt_students).intersection(audio_students)))

    consistencies = []

    for sid in tqdm(students):
        consistencies.append(check_consistency(
            os.path.join(audio_folder, str(sid), f"{sid}.mp3"),
            os.path.join(pt_folder, str(sid), f"{sid}_pyin_pitchtrack.csv"),
        ))
        
        
    print(np.mean(consistencies))
    
    
if __name__ == "__main__":
    import fire
    fire.Fire()
