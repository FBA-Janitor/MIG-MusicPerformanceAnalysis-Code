import glob
import os
from pprint import pprint
from typing import Optional, Sequence, Union
import warnings
from librosa import pyin as lpyin
from librosa import frames_to_time
import torch
import torchaudio
import yaml
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import soundfile as sf
import pandas as pd
from librosa import load as lload
import numpy as np
import pandas as pd

import pesto

from pesto.core import load_dataprocessor, load_model, reduce_activation, export

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



@torch.inference_mode()
def predict(
        x: torch.Tensor,
        sr: Optional[int] = None,
        model: Union[torch.nn.Module, str] = "mir-1k",
        data_preprocessor=None,
        step_size: Optional[float] = None,
        reduction: str = "argmax",
        chunk_size: Optional[int] = 4096,
        convert_to_freq: bool = False
):
    r"""Main prediction function.

    Args:
        x (torch.Tensor): input audio tensor,
            shape (num_channels, num_samples) or (batch_size, num_channels, num_samples)
        sr (int, optional): sampling rate. If not specified, uses the current sampling rate of the model.
        model: PESTO model. If a string is passed, it will load the model with the corresponding name.
            Otherwise, the actual nn.Module will be used for doing predictions.
        data_preprocessor: Module handling the data processing pipeline (waveform to CQT, cropping, etc.)
        step_size (float, optional): step size between each CQT frame in milliseconds.
            If the data_preprocessor is passed, its value will be used instead.
        reduction (str): reduction method for converting activation probabilities to log-frequencies.
        num_chunks (int): number of chunks to split the input audios in.
            Default is 1 (all CQT frames in parallel) but it can be increased to reduce memory usage
            and prevent out-of-memory errors.
        convert_to_freq (bool): whether predictions should be converted to frequencies or not.
    """
    # convert to mono
    assert 2 <= x.ndim <= 3, f"Audio file should have two dimensions, but found shape {x.size()}"
    batch_size = x.size(0) if x.ndim == 3 else None
    x = x.mean(dim=-2)

    if data_preprocessor is None:
        assert step_size is not None, \
            "If you don't use a predefined data preprocessor, you must at least indicate a step size (in milliseconds)"
        data_preprocessor = load_dataprocessor(step_size=step_size / 1000., device=x.device)

    # If the sampling rate has changed, change the sampling rate accordingly
    # It will automatically recompute the CQT kernels if needed
    data_preprocessor.sampling_rate = sr

    if isinstance(model, str):
        model = load_model(model, device=x.device)

    model = model.to(x.device)

    # apply model
    cqt = data_preprocessor(x)
    try:
        activations = torch.cat([
            model(chunk) for chunk in cqt.split(chunk_size)
        ])
    except torch.cuda.OutOfMemoryError:
        raise torch.cuda.OutOfMemoryError("Got an out-of-memory error while performing pitch estimation. "
                                          "Please increase the number of chunks with option `-c`/`--chunks` "
                                          "to reduce GPU memory usage.")

    if batch_size:
        total_batch_size, num_predictions = activations.size()
        activations = activations.view(batch_size, total_batch_size // batch_size, num_predictions)

    # shift activations as it should (PESTO predicts pitches up to an additive constant)
    activations = activations.roll(model.abs_shift.cpu().item(), dims=-1)

    # convert model predictions to pitch values
    pitch = reduce_activation(activations, reduction=reduction)
    if convert_to_freq:
        pitch = 440 * 2 ** ((pitch - 69) / 12)

    # for now, confidence is computed very naively just based on volume
    confidence = cqt.squeeze(1).max(dim=1).values.view_as(pitch)
    conf_min, conf_max = confidence.min(dim=-1, keepdim=True).values, confidence.max(dim=-1, keepdim=True).values
    confidence = (confidence - conf_min) / (conf_max - conf_min)

    timesteps = torch.arange(pitch.size(-1), device=x.device) * data_preprocessor.step_size

    return timesteps, pitch, confidence, activations


def run_pesto(
    wav_path,
    output_path,
    fs=22050,
    model_name="mir-1k",
    no_convert_to_freq=True,
    step_size: float = 10.,
    reduction: str = "alwa",
    export_format: Sequence[str] = ("csv",),
    gpu=0,
    chunk_size=2**14
):
    try:
        if gpu >= 0 and not torch.cuda.is_available():
            warnings.warn("You're trying to use the GPU but no GPU has been found. Using CPU instead...")
            gpu = -1
        device = torch.device(f"cuda:{gpu:d}" if gpu >= 0 else "cpu")

        # define data preprocessing
        data_preprocessor = load_dataprocessor(step_size / 1000., device=device)

        # define model
        model = load_model(model_name, device=device)#.to(device)
        predictions = None

            # load audio file
        x, sr = torchaudio.load(wav_path)

        if sr != fs:
            x = torchaudio.functional.resample(x, sr, fs)
            sr = fs

            
        x = x.to(device)

        
        x = x.to(device)

        # compute the predictions
        predictions = predict(x, sr, model=model, data_preprocessor=data_preprocessor, reduction=reduction,
                                convert_to_freq=not no_convert_to_freq, chunk_size=chunk_size)

        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        predictions = [p.cpu().numpy() for p in predictions]
        for fmt in export_format:
            export(fmt, output_path, *predictions)
    except Exception as e:
        print(e)

def process(
    audio_root="/media/fba/MIG-FBA-Audio/cleaned/audio/bystudent",
    pesto_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack_pesto",
):

    wav_paths = sorted(glob.glob(audio_root + "/**/*.wav", recursive=True))
    out_paths = [wav_path.replace(audio_root, pesto_root).replace('.wav', '') for wav_path in wav_paths]

    process_map(run_pesto, wav_paths, out_paths, max_workers=3, chunksize=1, total=len(wav_paths))

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