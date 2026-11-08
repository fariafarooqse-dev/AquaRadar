# Model ↔ UI contract (agree before integration)

The following is a proposed interface contract, not a claim that model inference is already implemented.

## UI sends
- Uploaded file bytes
- Original filename
- Optional location label
- Optional operator notes

## Model service returns
- `status`: `completed`, `uncertain`, or `error`
- `predicted_class`: one of the dataset-supported class labels, or null
- `confidence`: optional numeric score; show only if its meaning is documented and appropriate
- `model_version`: version/name of model used, or null
- `message`: plain-language summary
- `warnings`: list of validation or quality warnings

## Rules
- Never make up a prediction if model inference fails.
- Never treat confidence as a calibrated probability unless calibration has been tested.
- If the recording is unsupported, corrupted, empty, too short, or poor quality, return an error/uncertain result with an explanation.
- Do not add a noise class unless the training data supports it.
- Model prediction is a screening result, not a confirmed leak.
- Do not claim precise localization or water-loss estimates.
- Keep model preprocessing identical between training and inference.


## Implemented behaviour (model v1: `leak_model_v1`)

Implemented in `model_service.analyze_audio_bytes(file_bytes, filename)`.

- `predicted_class` is one of `leak`, `no_leak`, `noise` (the three folders of the training dataset), or `null`.
- `status="completed"`: top averaged score >= 0.60. `predicted_class` and `confidence` are set.
- `status="uncertain"`: top averaged score < 0.60. `predicted_class` is `null`; `confidence` holds the top score. Show "review manually".
- `status="error"`: empty file, unsupported suffix (`.wav`, `.flac`, `.ogg`), file over 25 MB, undecodable audio, shorter than 1 second, silent, or model not connected. `predicted_class`, `confidence` and `model_version` are `null`.
- `confidence` is the top random-forest vote share averaged over 1-second windows. It is NOT a calibrated probability. Label it "model score", never "probability of a leak".
- Recordings are cut into 1-second windows (first 60 s at most) and the window scores are averaged. This averaging was not evaluated separately.
- `warnings` always includes a screening-only notice. Show it.
- The 0.60 threshold was chosen from out-of-fold results on the same data, so it is optimistic.
- `no_leak` means "resembles the dataset's no-leak examples"; it does not rule out a leak. `noise` covers only the 15 sound categories in the dataset.
