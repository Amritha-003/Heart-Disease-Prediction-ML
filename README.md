# Heart Disease Prediction ML

An educational machine-learning example demonstrating data preparation, model training, and evaluation for a binary classification task.

## Important limitation

This repository currently uses a **small synthetic demonstration dataset**, not real patient data. The model is not clinically validated and must not be used to diagnose disease, estimate an individual's actual risk, or make medical decisions.

## Features

- Example tabular data with age, resting blood pressure, cholesterol, and maximum heart rate
- Train/test split
- Feature scaling and logistic regression pipeline
- Accuracy and classification report

## Requirements

- Python 3.9 or later
- Packages listed in `requirements.txt`

## Run locally

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the script:

   ```bash
   python heart_disease_prediction.py
   ```

## Next improvements

- Replace the toy data with a properly licensed, documented dataset.
- Use cross-validation and class-appropriate metrics.
- Check data quality, missing values, class imbalance, and possible bias.
- Keep a clear record of dataset source and model limitations.

## Disclaimer

This is a student learning project, not a medical device or clinical decision-support system.
