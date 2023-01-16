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

The `2023_happy_new_year_model.pth` is trained in Jan 2023.