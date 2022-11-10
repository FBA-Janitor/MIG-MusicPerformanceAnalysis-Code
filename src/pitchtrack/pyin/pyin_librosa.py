from librosa import pyin as lpyin
from librosa import note_to_hz
import yaml


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


def pyin(x, fs, config_path="/media/fba/MIG-MusicPerformanceAnalysis-Code/src/pitchtrack/configs/default.yaml", **kwargs):

    configs = read_yaml_to_dict(config_path)['pyin']

    return lpyin(x, sr=fs, **kwargs, **configs)
