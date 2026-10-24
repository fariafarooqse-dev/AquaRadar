# AquaRadar

AI-assisted acoustic screening and maintenance coordination prototype for water-leak monitoring.

> **Prototype limitation:** AquaRadar flags audio that resembles labelled leak-related sounds. It does not confirm a real leak, guarantee exact location, or estimate water loss. All demo/sample results must be labelled as simulated.

## Quick start

1. Install Python 3.10 or 3.11.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the starter interface:
   ```bash
   streamlit run app.py
   ```

The initial interface uses clearly labelled **demo results**. Do not present demo predictions as real model outputs.

## Planned architecture

- `app.py`: Streamlit UI and page navigation
- `audio_features.py`: audio loading, validation, preprocessing, feature extraction
- `train_model.py`: model training and evaluation
- `model_service.py`: stable interface between UI and trained model
- `database.py`: SQLite persistence for analyses and alerts
- `tests/`: basic checks
- `docs/`: product contract, UI requirements, and research notes

## Team ownership

- Member 1: model, feature pipeline, model service contract, product decision logic, integration and evaluation.
- Member 2: Streamlit UI/UX, dashboard, upload/results experience, alert-management screens; use demo data until Member 1 confirms the model contract.
- Member 3: market research, competitor research, user validation, branding, pitch and business model hypotheses.

## Git workflow

- Keep `main` runnable.
- Create branches for work, e.g. `ai-model`, `interface`, `market-research`.
- Make small commits with descriptive messages.
- Never commit passwords, API keys, private data, virtual environments, or large datasets.
- Check dataset licences before downloading or redistributing data.
