# AI Toxic Text Classifier

## Overview
This project demonstrates an end-to-end AI workflow:
- Data collection
- Data annotation (manual labeling)
- Data structuring (JSON)
- Model training and evaluation

## Dataset
- 100+ labeled text samples
- Categories: `toxic` and `clean`
- Stored in JSON format

## Technologies Used
- Python
- scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## Results
- Achieved ~80% accuracy on test data

## How to Run

```bash
pip install -r requirements.txt
python train.py
