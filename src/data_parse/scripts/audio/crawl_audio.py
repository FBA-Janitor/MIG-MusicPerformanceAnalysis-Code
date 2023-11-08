import os
from types import SimpleNamespace
import urllib.request

import pandas as pd

from utils.default_configs_path import xlsx_config_path
from utils.utils import read_yaml_to_dict

from tqdm import tqdm

import soundfile as sf

from tqdm.contrib.concurrent import process_map

def download_one(args):
    src = args.src
    dst = args.dst

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    try:
        urllib.request.urlretrieve(src, dst)
        args.success = True
        return args
    except Exception as e:
        print(f"Error downloading {src}")
        print(e)
        args.success = False
        return args

def process_year_band(
        root,
        year,
        band,
        output_root="MIG-FBA-Audio/new_download/audio/bystudent"
):

    xlsx_config = read_yaml_to_dict(root, xlsx_config_path)

    xlsx_path = xlsx_config["assessment_scores"][year][band]
    xlsx = xlsx_path["excel"]
    sheet = xlsx_path["sheet"]

    df = pd.read_excel(os.path.join(root, xlsx), sheet)

    recording_paths = df[["Student", "RecordingPath"]].drop_duplicates().set_index("Student")["RecordingPath"]

    print(f"Found {len(recording_paths)} recordings for {year} {band}")

    # print(recording_paths.head())

    output_root = os.path.join(root, output_root, str(year), band)

    inout = [
        SimpleNamespace(src=path, dst=os.path.join(output_root, f"{student}.mp3"))
        for student, path in recording_paths.items()
    ]

    successes = process_map(
        download_one,
        inout,
        chunksize=2,
        desc=f"Downloading {year} {band} recordings"
    )

    failed = [s for s in successes if not s.success]

    print(f"Failed to download {len(failed)} recordings for {year} {band}")

    with open(os.path.join(output_root, "failed.txt"), "w") as f:
        for fail in failed:
            f.write(f"{fail.src}\n")
    


if __name__ == "__main__":
    import fire
    fire.Fire()

    

