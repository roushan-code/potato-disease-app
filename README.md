# Potato Leaf Disease App

This Streamlit app uses the trained EfficientNetB2 model to classify potato-leaf images.

## Setup

1. Download `potato_disease_efficientnetb2.keras` from Kaggle.
2. Place it in this folder beside `app.py`.
3. Create and activate a Python environment.
4. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run locally

```powershell
streamlit run app.py
```

The browser will open the upload interface automatically.

## Important

The model expects the same seven classes used during training:

- Bacteria
- Fungi
- Healthy
- Nematode
- Pest
- Phytopthora
- Virus

The prediction is an AI estimate. It should not be treated as a definitive agricultural diagnosis.
