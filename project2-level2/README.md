# Wine Quality Test Project

## Project Overview
This project analyzes the relationship between various chemical properties of wine and their impact on perceived quality. Using a dataset of 1,143 wines, the project explores how different attributes like acidity, sugar content, alcohol percentage, and sulfur dioxide levels affect wine quality ratings.

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Key Findings](#key-findings)
4. [Usage](#usage)
5. [Model Performance](#model-performance)
6. [Contributing](#contributing)

## Project Description
The Wine Quality Test project uses machine learning to predict wine quality based on physicochemical properties. The analysis includes:
- Exploratory data analysis of 11 wine attributes
- Visualization of relationships between chemical properties and quality ratings
- A Random Forest Classifier model for quality prediction

## Dataset
The dataset contains 1,143 samples with 13 features (12 chemical properties + ID):

**Features:**
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable, scale 3-8)
- Id

**Data Characteristics:**
- No missing values
- No duplicates
- All numerical features

## Key Findings

### Chemical Properties vs. Quality

1. **Fixed Acidity**
   - Optimal range: 7-8
   - Quality declines at both low (<6) and high (>12) acidity levels

2. **Volatile Acidity**
   - Strong negative correlation with quality
   - Best quality at levels <0.5

3. **Citric Acid**
   - Negative correlation with quality
   - Best quality at low levels (0.0-0.2)

4. **Residual Sugar**
   - Negative correlation with quality
   - Best quality at levels <4 g/L

5. **Alcohol Content**
   - Optimal range: 9-10%
   - Quality declines sharply above 12%

6. **Sulfur Dioxide (Free & Total)**
   - Negative correlation with quality
   - Best quality at free SO₂ <20 mg/L and total SO₂ <50 mg/L

**Requirements:**
- Python 3.7+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

## Usage

1. Run the analysis notebook:
   ```bash
   jupyter notebook WineQualityAnalysis.ipynb
   ```

2. To make predictions using the trained model:
   ```python
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   import joblib

   # Load the trained model
   model = joblib.load('wine_quality_model.pkl')

   # Prepare input data (example values)
   input_data = pd.DataFrame([[7.4, 0.00, 0.00, 1.0, 0.06, 1.0, 34.0, 0.98, 2, 0.56, 2.0]],
                           columns=["fixed acidity", "volatile acidity", "citric acid",
                                   "residual sugar", "chlorides", "free sulfur dioxide",
                                   "total sulfur dioxide", "density", "pH",
                                   "sulphates", "alcohol"])

   # Make prediction
   prediction = model.predict(input_data)
   print(f"Predicted quality: {prediction[0]}")
   ```

## Model Performance
The Random Forest Classifier achieved an accuracy of **64.2%** on the test set.

**Potential Improvements:**
- Feature engineering to highlight important relationships
- Hyperparameter tuning
- Trying different model architectures
- Addressing class imbalance in quality ratings

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request
