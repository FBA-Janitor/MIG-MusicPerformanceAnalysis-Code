import glob
import os
from tqdm import tqdm
import tensorflow as tf
from .newcore import build_and_load_model, process_file

# tf.config.experimental.set_memory_growth(
#     tf.config.experimental.list_physical_devices("GPU")[0], True
# )


def process(
    audio_root="/media/fba/MIG-FBA-Audio/cleaned/audio/bystudent",
    f0_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack3",
    use_multiprocessing=False,
    model_capacity="full",
    min_year=2013,
    max_year=2018,
    smooth=True
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
    f0_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack3",
    use_multiprocessing=False,
    model_capacity="full",
    smooth=True
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
    smooth=True
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


if __name__ == "__main__":
    import fire

    fire.Fire()
