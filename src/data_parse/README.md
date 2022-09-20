# Data parsing and organization code

## List of Instruments
```
['Percussion',
 'Euphonium',
 'Trumpet',
 'Bb Clarinet',
 'French Horn',
 'Alto Saxophone',
 'Flute',
 'Trombone',
 'Oboe',
 'English Horn',
 'Bassoon',
 'Tuba',
 'Tenor Saxophone',
 'Bari Saxophone',
 'Bass Clarinet',
 'Bb Contrabass Clarinet',
 'Piccolo',
 'Eb Clarinet',
 'Eb Contra Alto Clarinet',
 'Bass Trombone',
 'Contrabassoon',
 'Soprano Sax',
 'Piano']
```

## General Guidelines

The project folder organization should look as follows:
```
.
+-- fba-canonical  
+-- MIG-FbaData    
+-- MIG-MusicPerformanceAnalysis-Code  
    +-- src
        +-- data_parse
            +-- README.md # <-- You are here
+-- MIG-FBA-DataCleaning   
    +-- canonical
    +-- cleaned           
+-- MIG-FBA-PitchTracking              
```

- `fba-canonical` is the folder from SharePoint, rename or use symlink if needed
- `MIG-FbaData` is the folder from MIG-FbaData.zip **not** GitHub
- `MIG-MusicPerformanceAnalysis-Code` is a fork of the GitHub Repo `GTCMT/MIG-MusicPerformanceAnalysis`, cleaning code goes here. **This is the current Repo**.
- `MIG-FBA-DataCleaning` is a fork of the GitHub Repo `GTCMT/MIG-FbaData`, cleaned symlinks go here. Clone [here](https://github.com/FBA-Janitor/MIG-FBA-Data-Cleaning).
    - `canonical` contains old files from the original repo. Nothing should be changed in this folder.
    - `cleaned` is where new symlinks should go. See [Folder Structure](#folder-structure).
- `MIG-FBA-PitchTracking` is a new repo for storing pitch track data. Clone [here](https://github.com/FBA-Janitor/MIG-FBA-PitchTracking).

**ALWAYS USE RELATIVE PATHS!**
For YAML files, use relative path relative to the top level project folder. E.g., this file is 
```
./MIG-MusicPerformanceAnalysis-Code/src/data_parse/README.md
```
regardless of where the code using that YAML may be located at. This allows the YAML to also be used in other codes easily.



## Folder Structure

Most folder structure in any cleaned repo with a `bystudent` folder should essentially follow the format below
```
cleaned
└── <data_type>
        └── bystudent
            └── <year>
                └── <band_type>
                    └── <instrument>
                        └── <student_id>
                            └── filename.ext
```

- `year` is the first year in the AY. If the AY is 2013-2014. `year` should be 2013.
- `band_type` takes either `middle`, `concert`, or `symphonic`.

### Assessment Scores (Karn)

```
cleaned
├── scores
│   └── summary
│       ├── maximum_scores.csv
│       └── <year>_<band_type>.csv
```

### Audio (Karn)

```
cleaned
├── audio
│       ├── 2013  
│       │   ├── concert
│       │   │   ├── Alto Saxophone
│       │   │   │   ├── 28667.json
│       │   │   │   └── ...
│       │   │   └── ...
│       │   └── ...
│       ├── <year>
│       │   └── <band_type>
│       │       └── <instrument>
│       │            └── <student_id>.json
│       └── ...
...
```


### Segmentation (Nikhil/Pavan)
```
cleaned
├── segmentation
│       ├── <year>
│       │   └── <band_type>
│       │       └── <instrument>
│       │            ├── <student_id>_instrument.txt
│       │            ├── <student_id>_segment.txt
│       │            └── <student_id>_seginst.csv
│       └── ...
...
```

- `*_seginst.csv` (working name, can be changed) is the combined csv of `*_instrument.txt` and `*_segment.txt`.
- `*_instrument.txt` and `*_segment.txt` are canonical and should be symlinked from `../canonical`.

### Pitch Tracking (Nikhil/Pavan)
```
/MIG-FBA-PitchTracking/cleaned
├── pitch_track
│       ├── <year>
│       │   └── <band_type>
│       │       └── <instrument>
│       │            ├── <student_id>_instrument.txt
│       │            ├── <student_id>_segment.txt
│       │            └── <student_id>_seginst.csv
│       └── ...
...
```

- Note that pyin data should be hosted in a separate repo `MIG-FBA-PitchTracking` due to thier combined size.
