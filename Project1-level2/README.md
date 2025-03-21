# Housing Price Prediction Model Documentation

## Overview
This project aims to predict housing prices based on various features such as the number of bedrooms, bathrooms, stories, and other amenities. The dataset contains 545 entries with 13 features. The project involves data exploration, visualization, and the creation of a predictive model using linear regression.

## Table of Contents
1. [Project Setup](#project-setup)
2. [Data Exploration](#data-exploration)
3. [Data Visualization](#data-visualization)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Feature Importance](#feature-importance)
6. [Conclusion](#conclusion)

## Project Setup

### Dataset
The dataset consists of 545 entries with 13 attributes:
- `price`: The price of the house (target variable)
- `area`: The size of the house in square feet
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `stories`: Number of stories
- `mainroad`: Whether the house is on the main road (yes/no)
- `guestroom`: Availability of a guestroom (yes/no)
- `basement`: Whether the house has a basement (yes/no)
- `hotwaterheating`: Availability of hot water heating (yes/no)
- `airconditioning`: Whether the house has air conditioning (yes/no)
- `parking`: Number of parking spaces
- `prefarea`: Whether the house is in a preferred area (yes/no)
- `furnishingstatus`: The furnishing status (furnished/semi-furnished/unfurnished)
  
### Libraries Used
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Seaborn**: For advanced data visualization.
- **Scikit-learn**: For machine learning model creation and evaluation.

### Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
```

### Loading the Dataset
```python
data_path = "C:/Users/Alimi Nimotalahi/Desktop/Oasis Infobyte/Housing/Housing.csv"
data = pd.read_csv(data_path)
data.head()
```

## Data Exploration

### Dataset Information
- **Number of Rows**: 545
- **Number of Columns**: 13
- **Columns**: `price`, `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus`

### Checking for Duplicates and Null Values
```python
data.duplicated().sum()  # No duplicates
data.isnull().sum()  # No null values
```

### Statistical Summary
```python
data.describe()
```

## Data Visualization

### How Does the Number of Bedrooms Affect House Price?
```python
no_bedroom_to_house_price = data.groupby("bedrooms")["price"].sum().reset_index()
sns.barplot(x="bedrooms", y="price", data=no_bedroom_to_house_price, color="orange")
```
**Observation**: Apartments with 3 bedrooms sell more.

### How Does the Number of Bathrooms Affect House Price?
```python
no_bathroom_to_house_price = data.groupby("bathrooms")["price"].sum().reset_index()
sns.barplot(x="bathrooms", y="price", data=no_bathroom_to_house_price, color="b")
```
**Observation**: Apartments with one bathroom sell more.

### How Does the Number of Stories Affect House Price?
```python
no_stories_to_house_price = data.groupby("stories")["price"].sum().reset_index()
sns.barplot(x="stories", y="price", data=no_stories_to_house_price)
```
**Observation**: Apartments with 2 stories sell more.

### How Does Mainroad Affect House Price?
```python
mainroad_to_house_price = data.groupby("mainroad")["price"].sum().reset_index()
sns.barplot(x="mainroad", y="price", data=mainroad_to_house_price, color="orange")
```
**Observation**: Apartments on the main road sell more.

### How Does Guestroom Affect House Price?
```python
guestroom_to_house_price = data.groupby("guestroom")["price"].sum().reset_index()
sns.barplot(x="guestroom", y="price", data=guestroom_to_house_price)
```
**Observation**: Apartments with no guest room sell more.

### How Does Basement Affect House Price?
```python
basement_to_house_price = data.groupby("basement")["price"].sum().reset_index()
sns.barplot(x="basement", y="price", data=basement_to_house_price, color="blue")
```
**Observation**: Apartments with no basement sell more.

### How Does Hot Water Heating Affect House Price?
```python
hotwaterheating_to_house_price = data.groupby("hotwaterheating")["price"].sum().reset_index()
sns.barplot(x="hotwaterheating", y="price", data=hotwaterheating_to_house_price, color="orange")
```
**Observation**: Apartments with no hot water heating sell more.

### How Does Air Conditioning Affect House Price?
```python
airconditioning_to_house_price = data.groupby("airconditioning")["price"].sum().reset_index()
sns.barplot(x="airconditioning", y="price", data=airconditioning_to_house_price)
```
**Observation**: Apartments with no air conditioning sell more.

### How Does Parking Space Affect House Price?
```python
parking_to_house_price = data.groupby("parking")["price"].sum().reset_index()
sns.barplot(x="parking", y="price", data=parking_to_house_price, color="green")
```
**Observation**: Apartments with no parking space sell more.

### How Does Preferred Area Affect House Price?
```python
prefarea_to_house_price = data.groupby("prefarea")["price"].sum().reset_index()
sns.barplot(x="prefarea", y="price", data=prefarea_to_house_price, color="orange")
```
**Observation**: Apartments not in preferred areas sell more.

### How Does Furnishing Status Affect House Price?
```python
furnishingstatus_to_house_price = data.groupby("furnishingstatus")["price"].sum().reset_index()
sns.barplot(x="furnishingstatus", y="price", data=furnishingstatus_to_house_price, color="green")
```
**Observation**: Semi-furnished apartments sell more.

## Model Training and Evaluation

### Data Preparation
```python
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
data = data.drop(columns=categorical_cols)
data = pd.concat([data, encoded_df], axis=1)

X = data.drop(columns="price")
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Linear Regression Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
```
**Mean Squared Error**: 1355172428958.8293

### Lasso Regression Model
```python
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': abs(lasso.coef_)})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)
```

## Feature Importance
The Lasso regression model was used to determine the importance of each feature in predicting house prices. The most important features are:
1. **Bathrooms**
2. **Air Conditioning (No)**
3. **Hot Water Heating (No)**
4. **Preferred Area (No)**
5. **Stories**

## Dependencies
Ensure the following libraries are installed before running the scripts:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Results
- Houses with 3 bedrooms, 1 bathrooms, and 2 stories tend to have higher prices.
- Houses located on the main road sell for higher prices.
- Semi-furnished houses are more valuable than fully furnished or unfurnished ones.

## Conclusion
This project successfully explored the factors affecting house prices and built a predictive model using linear regression. The model achieved a mean squared error of 1355172428958.8293, and feature importance analysis highlighted the most significant predictors of house prices. Future work could involve experimenting with more advanced models and hyperparameter tuning to improve prediction accuracy.

## Future Work
- Experimenting with advanced machine learning models such as Decision Trees and Random Forest.
- Exploring additional feature engineering techniques to enhance model performance.

## Author
**Alimi Afeez Olalekan**


