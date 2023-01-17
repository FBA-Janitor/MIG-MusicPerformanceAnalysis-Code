# Auto Segmentation

The code is for generation of automatic segmentation and the segments will go into later music performance analysis code.
*Note: the code does not share the same dataloader as the performance analysis code.*

## Usage
### Generate segmentation
To generate segmentation, run

```
python -m fire segment_wav.py segment_dir --input_dir <input_dir> --output_dir <output_dir>
```

The `input_dir` should be a directory containing audio pieces, and the folder structure will be `input_dir/stu_id/stu_id.mp3`. All the audio will be processed. You might want to change the temporay folder to save the feature and the model to use in [default config path](scripts/utils/default_configs_path.py).

If the feature has already been extracted, you can also run

```
python -m fire segment_wav.py segment_dir --input_dir <input_dir> --output_dir <output_dir> --from_feature True
```

The `input_dir` should be a directory containing extracted features, and the folder structure will be `input_dir/stu_id.npz`.

### Train a new SVM model

To train a new SVM model, you should first generate the `csv` files of training data by running

```
python -m fire generate_audio_segment.py generate_multi_year --output_dir <output_dir> --root_audio_dir <audio_dir> --root_segment_dir <segment_dir>
```

- `output_dir` is the directory to save the `csv` files
- `audio_dir` is the audio directory of the data repo
- `segment_dir` is the segmentation directory of the data repo

Then to train the model, run

```
python -m fire train_svm.py train_svm
```

You might want to change the training data, the segment status file, the temporary feature folder and the model save pth in [default config path](scripts/utils/default_configs_path.py)

## Model details

### 2023_happy_new_year_model.pth
The `2023_happy_new_year_model.pth` is trained in Jan 2023.

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

| Test data    | Seg1  | Seg2  | Seg3  | Seg4  | Seg5  | Piece |
| :----------: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2013 concert | 96.75 | 94.02 | 90.61 | 96.88 | 96.87 | 97.65 |
| 2014 concert | 96.75 | 93.34 | 89.82 | 96.36 | 96.76 | 97.41 |
| 2015 concert | 96.78 | 93.69 | 90.03 | 97.36 | 96.77 | 97.43 |