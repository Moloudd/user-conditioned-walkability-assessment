# Data

- **Sidewalk-view imagery:** Sourced from [footpath.ai](https://footpath.ai)
  and permitted for public release. The imagery is available through the
  [`mdamandeh/user-conditioned-walkability-assessment`](https://huggingface.co/datasets/mdamandeh/user-conditioned-walkability-assessment)
  dataset on Hugging Face. Download the dataset and place the image files
  under `data/images/` using the directory structure shown below.

- **Survey data:** Participant demographics and individual walkability
  perception ratings for the sidewalk-view images will be made publicly
  available and linked here.

## Expected layout

Scripts in `scripts/` read from `$WALKABILITY_DATA_DIR` (default: `./data`,
see `.env.example`), expecting:

```
data/
├── images/                  # sidewalk-view images referenced by `image_path`
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

## Expected CSV schema

Each split CSV must contain:

| column              | type          | notes                                              |
|---------------------|---------------|-----------------------------------------------------|
| `image_path`         | str           | path to an image file, relative to `images/`        |
| `rating`             | int (1–5)     | ordinal walkability rating — the prediction target  |
| `response_id`        | str           | survey response identifier (per-user grouping key)  |
| `age`                | int (ordinal) | respondent attributes feature                |
| `gender`             | int (categorical) | respondent attributes feature                       |
| `childhood_country`  | int (categorical) | respondent attributes feature                       |
| `childhood_area`     | int (categorical) | respondent attributes feature                       |
| `disability`         | int (categorical) | respondent attributes feature                       |
| `walking_frequency`  | int (ordinal) | respondent attributes feature                       |
| `residence_type`     | int (categorical) | respondent attributes feature                       |
