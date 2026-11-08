# Dataset documentation — Zenodo record 18631450

Last updated: 2026-10-25. Sources: the saved Zenodo record metadata (`README_zenodo.txt`, "Record") and
AquaRadar's own inspection scripts (`scripts/inspect_dataset.py`, `scripts/build_manifest_draft.py`,
`scripts/make_grouped_folds.py`), marked "Measured". Nothing below is a claim about field performance.

## 1. Source and licence (Record)

| Item | Value |
|---|---|
| Title | Acoustic data for the manuscript entitled 'Self-supervised acoustic leakage detection for water distribution systems: A real-time diagnosis framework under data scarcity' |
| Authors (as listed) | WANG, QI (contact person); Mei, Zhongyi (data manager); Zhan, Fan and Chen, Jiongxi (data collectors) |
| Affiliation | Guangdong University of Technology |
| Version / date | v1, published 2026-02-13 |
| DOI / URL | https://doi.org/10.5281/zenodo.18631450 · https://zenodo.org/records/18631450 |
| Funding listed | Guangzhou Municipal Science and Technology Bureau, award 2023B03J1333 |
| Licence | Creative Commons Attribution 4.0 International (`cc-by-4.0`), https://creativecommons.org/licenses/by/4.0/legalcode |
| Access | Open; 3 files (`.rar`), 5,651,250 bytes in total |

**Attribution to use** (CC BY requires credit, a link to the licence, and an indication of changes):

> Wang, Q., Mei, Z., Zhan, F., & Chen, J. (2026). *Acoustic data for the manuscript entitled
> 'Self-supervised acoustic leakage detection for water distribution systems: A real-time diagnosis framework
> under data scarcity'* (v1). Zenodo. https://doi.org/10.5281/zenodo.18631450. Licensed under CC BY 4.0.

**Changes made by AquaRadar:** extracted 39 audio features per clip (feature set f1); excluded 36 byte-identical
duplicate files from training and evaluation; trained a random-forest classifier. The dataset authors do not
endorse AquaRadar. The raw audio is not redistributed in this repository (`data/raw/` is git-ignored). This
document is not legal advice.

## 2. What the record states

- 1,000 one-second audio clips, "mostly collected from an outdoor leakage detection training base located at
  Dongguan in Southern China". The authors say the data were used to train deep-learning leak-detection models.
- `leak` folder: 500 clips. `no leak` folder: 386 clips. Both were "collected on-site from the leakage detection
  training base". Filename rule: Pipe Material-Region-Pressure (MPa)-Flow Rate (m/s)-Collection Device; missing
  values are `NA`.
- `environmental noise` folder: 114 clips, described by the authors as having "no-leak labels". Filename rule:
  Noise Category-Region-Collection Device. "Most" were downloaded from https://www.dlmeasure.com/ and "a small
  portion" were collected on-site.

## 3. What the record does NOT state (unverified)

- Sampling rate, microphone/hydrophone/logger models, and recording settings.
- How leak and no-leak labels were verified, or whether leaks were artificial.
- Whether clips are slices of longer recordings, and what the `0-1`, `4-5` filename tokens mean.
- What the "Region" values mean, and whether pressure and flow values were measured per clip.
- The licence status of the downloaded noise clips on the third-party website (the record applies CC BY 4.0
  to the whole record; AquaRadar has not checked the source site's terms).

## 4. Measured by AquaRadar

- Files: 1,000 `.wav` plus one non-audio `.ini` file (presumed Windows `desktop.ini`; not confirmed).
- Format: all 1,000 readable, mono, 8,000 Hz, 16-bit PCM, exactly 1.00 s (0.28 hours in total).
- Class counts match the record: leak 500, no_leak 386, noise 114. Filenames parsed: 1,000 of 1,000.
- Duplicates: 36 groups (72 files) of byte-identical audio, all inside one folder; 4 groups have filenames that
  describe different recordings, so filename metadata is not fully reliable. After removing the extra copy of
  each: leak 496, no_leak 356, noise 112 (964 files).
- Noise categories (15): drilling, dog, engine, hydrant, rain, grass noise, fan motor, alarm, drum, fan,
  flowing water, gas pipe, unknown, wind, music. All noise files name "noise logger" as device.
- Metadata is associated with the label (all 1,000 files):
  - Device: leak = 391 noise logger / 107 hydrophone / 2 NA; no_leak = 279 / 107 / 0.
  - Material: leak = 281 ductile iron, 194 PE, 7 PVC, 12 steel, 6 NA; no_leak = 178, 175, 3, 10, 20.
  - Pressure is `NA` for all 386 no_leak files (and 31 of 500 leak files); flow is `NA` for all no_leak and 74 leak
    files. Pressure and flow must never be model inputs because they would reveal the label.
- Related clips: grouping by filename key plus identical audio gives 16 leak, 18 no_leak and 15 noise groups
  (after duplicate removal). The largest groups hold 140 leak, 66 no_leak and 25 noise files.

## 5. Risks and how they are handled

| Risk | Handling | Residual |
|---|---|---|
| Near-duplicate clips across train and test | Grouped 5-fold CV; audit shows no group or identical audio in two folds | Group definition is a filename-based hypothesis |
| Metadata shortcuts | `metadata_only` baseline run; audio models compared against it | Metadata still correlates with labels |
| Noise class = 15 sound types, partly third-party | Noise folds test unseen categories | Noise recall is weak; do not claim general noise rejection |
| Single site, 8 kHz, 1-second clips | Stated in model card and UI warnings | Not validated on campus pipes |
| Class balance | `class_weight="balanced"`; per-class metrics reported | Only 16–18 groups per pipe class |
