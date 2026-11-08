# \# AquaRadar

# 

# AI-assisted acoustic screening prototype for water-leak monitoring.

# 

# > \*\*Prototype limitation:\*\* AquaRadar flags audio that resembles labelled leak-related sounds. It does not confirm a real leak, guarantee exact location, or estimate water loss.

# 

# \## Quick start

# 

# 1\. Install Python 3.10 or 3.11.

# 2\. Create and activate a virtual environment.

# 3\. Install dependencies:

# &#x20;  ```bash

# &#x20;  pip install -r requirements.txt

# &#x20;  ```

# 4\. Run the interface:

# &#x20;  ```bash

# &#x20;  streamlit run app.py

# &#x20;  ```

# 5\. Run the tests:

# &#x20;  ```bash

# &#x20;  pytest

# &#x20;  ```

# 

# \## Project status

# 

# \*\*Built:\*\*

# \- Audio feature extraction (`audio\_features.py`)

# \- Leak / no-leak / noise classifier (`train\_model.py`, `model\_service.py`)

# \- Streamlit prototype (`app.py`)

# \- SQLite persistence (`database.py`)

# \- Automated tests (`tests/`)

# \- Feature-set comparison experiments (`Scripts/`, `outputs/`). Model results are in `outputs/final\_model\_report.txt`.

# 

# \*\*In progress:\*\*

# \- Hardware: piezo contact pickup with a pre-amplifier, recorded through a smartphone

# \- Recordings from a small controlled pipe setup, to build a localized dataset

# 

# \*\*Planned:\*\*

# \- Urdu-English interface

# \- Maintenance and inspection records

# \- Testing on real pipelines

# 

# \## Architecture

# 

# \- `app.py`: Streamlit UI

# \- `audio\_features.py`: audio loading, validation, feature extraction

# \- `train\_model.py`: model training and evaluation

# \- `model\_service.py`: interface between the UI and the trained model

# \- `database.py`: SQLite storage for analyses and alerts

# \- `tests/`: automated checks

# \- `docs/`: model–UI contract (more notes to be added)

# 

# \## Data

# 

# Training uses public leak-audio datasets, which are not included in this repository. Check each dataset's licence before downloading or redistributing it. Our own recordings will be added as they are collected.

# 

# \## Git workflow

# 

# \- Keep `main` runnable.

# \- Use small commits with descriptive messages.

# \- Never commit passwords, API keys, private data, virtual environments, or large datasets.

