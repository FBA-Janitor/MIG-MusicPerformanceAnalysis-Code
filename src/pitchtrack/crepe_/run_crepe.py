import glob
import os
from tqdm import tqdm
import tensorflow as tf
from .newcore import build_and_load_model, process_file
import numpy as np

# tf.config.experimental.set_memory_growth(
#     tf.config.experimental.list_physical_devices("GPU")[0], True
# )
import pandas as pd
from scipy import signal as sps

from tqdm.contrib.concurrent import process_map

def process(
    audio_root="/media/fba/MIG-FBA-Audio/cleaned/audio/bystudent",
    f0_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack3",
    use_multiprocessing=False,
    model_capacity="full",
    min_year=2013,
    max_year=2018,
    smooth=True,
):
    if "3" in f0_root and smooth:
        raise ValueError("Cannot smooth 3rd version of pitchtrack")

    wav_paths = sorted(glob.glob(audio_root + "/**/*.wav", recursive=True))

    wav_paths = [
        w
        for w in wav_paths
        if int(w.split("/")[-4]) >= min_year and int(w.split("/")[-4]) <= max_year
    ]

    # print(wav_paths)

    model = build_and_load_model(model_capacity)

    for w in tqdm(wav_paths):
        outpath = os.path.dirname(w.replace(audio_root, f0_root))
        os.makedirs(outpath, exist_ok=True)
        process_file(model, w, outpath, viterbi=smooth)


def process_2013_to_2015(
    audio_root="/media/fba/MIG-FBA-Audio/cleaned/audio/bystudent",
    f0_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack4",
    use_multiprocessing=False,
    model_capacity="full",
    smooth=True,
):
    if "3" in f0_root and smooth:
        raise ValueError("Cannot smooth 3rd version of pitchtrack")

    process(
        audio_root=audio_root,
        f0_root=f0_root,
        use_multiprocessing=use_multiprocessing,
        model_capacity=model_capacity,
        min_year=2013,
        max_year=2015,
    )


def process_2016_to_2018(
    audio_root="/media/fba/MIG-FBA-Audio/cleaned/audio/bystudent",
    f0_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack4",
    use_multiprocessing=False,
    model_capacity="full",
    smooth=True,
):

    if "3" in f0_root and smooth:
        raise ValueError("Cannot smooth 3rd version of pitchtrack")

    process(
        audio_root=audio_root,
        f0_root=f0_root,
        use_multiprocessing=use_multiprocessing,
        model_capacity=model_capacity,
        min_year=2016,
        max_year=2018,
    )


def process_symphonic_clarinet(
    audio_root="/media/fba/MIG-FBA-Audio/cleaned/audio/bystudent",
    f0_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack5",
    use_multiprocessing=False,
    model_capacity="full",
    smooth=True,
):

    splits = np.load("/home/kwatchar3/fba/split_info.npz", allow_pickle=True)
    train = splits["train"]
    val = splits["val"]

    wav_paths = [
        os.path.join(audio_root, year, band, sid, f"{sid}.wav")
        for sid, year, band in np.concatenate([train, val], axis=0)
    ]

    wav_paths = [w for w in wav_paths if os.path.exists(w)]

    # print(wav_paths)
    # wav_paths = sorted(glob.glob(audio_root + "/**/*.wav", recursive=True))

    # print(wav_paths)

    model = build_and_load_model(model_capacity)

    for w in tqdm(wav_paths):
        outpath = os.path.dirname(w.replace(audio_root, f0_root))
        if os.path.exists(outpath):
            continue
        os.makedirs(outpath, exist_ok=True)
        process_file(model, w, outpath, viterbi=smooth, step_size=256 / 44100 * 1000)


def resample_pyin(f0path):
    
    df = pd.read_csv(f0path).rename(columns={"Time": "time", "MIDI": "frequency"})

    f = df["frequency"].values
    t = df["time"].values
    f = np.nan_to_num(f, nan=0)

    num = int(np.round(t[-1]/(10 / 1000))) + 1

    fr, tr = sps.resample(f, num=num, t=t)

    df = pd.DataFrame({"time": tr, "frequency": fr})

    # print(df)

    os.makedirs(os.path.dirname(f0path.replace("pitchtrack", "pitchtrack_resampled")), exist_ok=True)

    df.to_csv(f0path.replace("pitchtrack", "pitchtrack_resampled"), index=False)



def resample_pyin_symcla(f0_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack"):
    splits = np.load("/home/kwatchar3/fba/split_info.npz", allow_pickle=True)
    train = splits["train"]
    val = splits["val"]

    f0_paths = [
        os.path.join(
            f0_root, "bystudent", year, band, sid, f"{sid}_pyin_pitchtrack.csv"
        )
        for sid, year, band in np.concatenate([train, val], axis=0)
    ]

    f0_paths = [f for f in f0_paths if os.path.exists(f)]

    process_map(resample_pyin, f0_paths, max_workers=16, chunksize=2)



if __name__ == "__main__":
    import fire

    fire.Fire()
