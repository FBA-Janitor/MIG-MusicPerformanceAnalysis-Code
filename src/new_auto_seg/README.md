# Auto Segmentation

The code is for generation of automatic segmentation and the segments will go into later music performance analysis code.
*Note: the code does not share the same dataloader as the performance analysis code.*

## Usage
### Generate segmentation
To generate segmentation, run

```
python segment_wav.py segment_dir --input_dir <input_dir> --output_dir <output_dir>
```

The `input_dir` should be a directory containing audio pieces, and the folder structure will be `input_dir/stu_id/stu_id.mp3`. All the audio will be processed. You might want to change the temporay folder to save the feature and the model to use in [default config path](scripts/utils/default_configs_path.py).

If the feature has already been extracted, you can also run

```
python segment_wav.py segment_dir --input_dir <input_dir> --output_dir <output_dir> --from_feature True
```

The `input_dir` should be a directory containing extracted features, and the folder structure will be `input_dir/stu_id.npz`.

### Train a new SVM model

To train a new SVM model, you should first generate the `csv` files of training data by running

```
python generate_audio_seg_path.py --output_dir <output_dir> --root_audio_dir <audio_dir> --root_segment_dir <segment_dir>
```

- `output_dir` is the directory to save the `csv` files
- `audio_dir` is the audio directory of the data repo. Default: `audio_dir` in [default config](scripts/utils/default_configs_path.py)
- `segment_dir` is the (manually labeled) segmentation directory of the data repo. Default: `segment_dir` in [default config](scripts/utils/default_configs_path.py)

Then to train the model, run

```
python fire train_svm.py train_svm
```

You might want to change the training data, the segment status file, the temporary feature folder and the model save pth in [default config path](scripts/utils/default_configs_path.py)

## Model details

### 2024_happy_new_year_model.pth
The `2024_happy_new_year_model.pth` is trained in Feb 2024.

The training uses the data below

- 2013 concert AltoSaxophone
- 2013 concert BbClarinet
- 2013 concert Flute
- 2013 middle AltoSaxophone
- 2013 middle BbClarinet
- 2013 middle Flute
- 2013 middle Trumpet
- 2013 middle Oboe
- 2014 concert AltoSaxophone
- 2014 concert BbClarinet
- 2014 concert Flute
- 2014 middle AltoSaxophone
- 2014 middle BbClarinet
- 2014 middle Flute

The evaluation results are below

| Test data                     | Seg1  | Seg2  | Seg3  | Seg4  | Seg5  | Piece | Success |
| :---------------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :-----: |
| 2015 middle  AltoSaxophone    | 94.56 | 88.52 | 87.15 | 95.46 | 96.40 | 96.22 | 120/122 |
| 2015 middle  BbClarinet       | 96.13 | 91.16 | 87.39 | 93.06 | 95.65 | 97.21 | 164/167 |
| 2015 middle  Flute            | 95.08 | 93.12 | 90.48 | 95.52 | 96.53 | 97.02 | 178/180 |
| 2016 concert AltoSaxophone    | 88.50 | 83.58 | 76.82 | 90.96 | 93.05 | 94.77 | 124/134 |
| 2016 middle  AltoSaxophone    | 91.88 | 84.43 | 80.51 | 91.91 | 93.79 | 93.79 | 111/111 |
| 2016 middle  BbClarinet       | 95.63 | 94.39 | 87.21 | 93.58 | 95.40 | 97.09 | 140/148 |

### 2024_happy_new_year_model.pth
The `2024_happy_new_year_model.pth` is trained in Feb 2023. It normalize the audio when read the audio so that it does not drive our advisor crazy.

## Updates
- Feb 04, 2024: Documentations updated. Functions renamed: now all the internal function names start with an underscore