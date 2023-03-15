import os
import yaml


def read_yaml_to_dict(root: str, path_relative_to_root: str) -> dict:
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
    with open(os.path.join(root, path_relative_to_root), "r") as stream:
        data = yaml.safe_load(stream)
    return data
