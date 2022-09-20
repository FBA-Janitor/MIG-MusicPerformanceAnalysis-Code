# Configuration Files for Assessment Score Processing


- [Configuration Files for Assessment Score Processing](#configuration-files-for-assessment-score-processing)
  - [`assessment_score_paths`](#assessment_score_paths)
  - [`max_score_replacements.yaml`](#max_score_replacementsyaml)
  - [`assesssment_score_name_changes.yaml`](#assesssment_score_name_changesyaml)
    - [Changes to instrument name](#changes-to-instrument-name)
    - [Changes to header names in the raw score data files](#changes-to-header-names-in-the-raw-score-data-files)
    - [Changes to assessment scores](#changes-to-assessment-scores)


## `assessment_score_paths`
This file contain a mapping to the path (relative to project top-level folder) where the excel files containing the assessment scores are stored.

Schema
```YAML
max_score: <path to file containing the maximum score info for all years>
assessment_scores:
    <year>: # int
        <band level>: # concert | middle | symphonic
            excel: <path to the excel file> # str
            sheet: <name of the excel sheet containing raw scores> # str
```
- Make sure that the `sheet` field is pointing to the worksheet with simple excel content and **NOT** PivotTable. 
- `max_score` field may have to change in the future if max scores depend on multiple files.

Example:
```YAML
max_scores: ./fba-canonical/All-State Bands Max Score.xlsx
assessment_scores:
  2013:
    concert:
      excel: ./fba-canonical/audio/2013-2014/Concert Band Scores.xlsx
      sheet: Raw Scores
    middle:
      excel: ./fba-canonical/audio/2013-2014/Middle School.xlsx
      sheet: Raw Scores
    symphonic:
      excel: ./fba-canonical/audio/2013-2014/Symphonic Band Scores.xlsx
      sheet: Raw Scores
  2014:
    concert:
      excel: ./fba-canonical/audio/2014-2015/Concert Band 2014-2015.xlsx
      sheet: Raw Data
    middle:
      excel: ./fba-canonical/audio/2014-2015/Middle School 2014-2015.xlsx
      sheet: Raw Data
    symphonic:
      excel: ./fba-canonical/audio/2014-2015/Symphonic Band 2014-2015.xlsx
      sheet: Raw Data
```

## `max_score_replacements.yaml`

Some of the `ScoreGroup` and `Description` fields in the excel file contain typos or inconsistencies that will fail with naive query. This file acts as a lookup table to search for the correct maximum score.

Schema
```YAML
missing: # seq of maps
    - Conditions: # map
        Year: [<years>] # optional, seq of int
        BandLevel: [<concert | middle | symphonic>] # optional, seq of (concert | middle | symphonic)
        ScoreGroup: [<score group name>] # optional, seq of str
        Description: [<description name>] # optional, seq of str
      ReplaceWith: # map
        Year: <replacement year, optional>
        ScoreGroup: <replacement score group, optional>
        Description: <replacement description, optional>
```

- **IMPORTANT**: The order of the maps within the `missing` sequence matters. In programmatic access, **ONLY** the first map whose conditions are satified will be applied.
- The `Conditions` maps
  - All fields in `Conditions` are optional but, if present, are chained together with logical `AND`s. 
  - Not specifying any of the subcondition will mean that **no restriction** applies to that level of search.
  - Within each subcondition, the value is a sequence chained together with logival `OR`s. Matching **any** of the item in the list will mean that subcondition is satisfied.
- The `ReplaceWith` maps
  - All fields in `ReplaceWith` are optional.
  - If the field is not present, no replacement is made for that field.

Example
```YAML
missing:
  # This replaces max scores for the score "Rhythmic Accuracy & Articulation" under the "Reading-Snare" group for concert and symphonics bands in year 2015, with a "Rhythmic Accuracy" score description and no change in score group or year.
  - Conditions: 
      Year: [2015]
      BandLevel: [concert, symphonic]
      ScoreGroup: [Reading-Snare]
      Description: [Rhythmic Accuracy & Articulation]
    ReplaceWith:
      Description: Rhythmic Accuracy
  
  # This replace ALL max scores for the year 2018 with the equivalent ones from 2017
  - Conditions:
      Year: [2018] 
    ReplaceWith:
      Year: 2017
```

## `assesssment_score_name_changes.yaml`
This file contains name changes made assessment score categories, instrument names, and data headers.

### Changes to instrument name

As of the time of writing, only one instrument has a name change, namely, Eb Clarinet. It was spelt as "EbClarinet" in 2013-2014.

Schema
```YAML
instrument:
    <old instrument name>: # str
        <new instrument name> # str
```

Example:
```YAML
instrument:
    EbClarinet: Eb Clarinet
```

### Changes to header names in the raw score data files
This is made mainly due to a header change in the excel files from 2014 onwards.

Schema
```YAML
columns:
    <year>: # int
        <band level>: # concert | middle | symphonic
            <old column name>: # str
                <new column name> # str
```

Example
```YAML
columns:
  2013:
    concert:
      CategoryGroup: ScoreGroup
      Category: Description
    middle:
      CategoryGroup: ScoreGroup
      Category: Description
    symphonic:
      CategoryGroup: ScoreGroup
      Category: Description
```

### Changes to assessment scores
This object documents changes in the score categories.

Schema
```YAML
assessments: # seq of maps
    - OldScoreGroup: <old score group name>             #str
      OldDescription: <optional, old description name>  #str
      NewScoreGroup: <optional, new score group name>   #str
      NewDescription: <optional, new description name>  #str
      Affects: # seq of maps
        - Year: <year affected>                 # int
          BandLevel: [<band level affected>]    # seq of (concert | middle | symphonic)
      Reason: Variant | Regroup | Similar | Legacy      # str
```


- **IMPORTANT**: The order of the maps within the `assessments` sequence matters. In programmatic access, the changes will be applied in the order it appears on the sequence. 
    - If an change item down the list is affecting a score that was affected by another change item before it, the identifiers used in the later items have to follow the changes already made by the previous change item. 
    - Generally, put more specific change items earlier in the list, and a blanket change later in the list.
- If `OldDescription` field is not present, the changes applied to the entire `OldScoreGroup` score group.
- If `NewScoreGroup` and/or `NewDescription` is not present, the respective field remains unchanged.
- `Reason` field identifies why a rename is made. 
    - `Variant` is used for typos and simple renaming where it is clear that the new and old names refer to the exact same musical concept
    - `Similar` is used when the old and new names may or may not share exactly the same musical concept depending on interpretation, but are similar enough. This tag is essentially used as a less confident version of `Variant`.
    - `Regroup` is used when the description is moved to another score group.
    - `Legacy` is used to mark score types that are no longer in used. When using this tag, the `Affects` map contains all the years and band levels this score used to apply to.


Example:
```YAML
assessments:
  - OldScoreGroup: Scales
    OldDescription: Chromatic
    NewScoreGroup: Chromatic Scale
    NewDescription: Chromatic Scale
    Affects:
      - Year: 2013
        BandLevel: [concert, middle, symphonic]
      - Year: 2014
        BandLevel: [concert, symphonic]
    Reason: Regroup

  - OldScoreGroup: Lyrical Etude
    OldDescription: Musicallity, Tempo & Style
    NewDescription: Musicality, Tempo, & Style
    Affects:
      - Year: 2013
        BandLevel: [concert, middle, symphonic]
      - Year: 2014
        BandLevel: [concert, middle, symphonic]
      - Year: 2015
        BandLevel: [concert, middle, symphonic]
      - Year: 2016
        BandLevel: [concert, middle, symphonic]
      - Year: 2017
        BandLevel: [concert, middle, symphonic]
      - Year: 2018
        BandLevel: [concert, middle, symphonic]
    Reason: Variant
```
