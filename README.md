##  Student Academic Performance Prediction (Multilinear Regression)

This project predicts student academic performance based on academic, behavioral, and lifestyle factors using a Multilinear Regression model.

## Dataset
**Source:** Kaggle – Students Performance in Exams  
(https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

## Objectives
- Load student performance data into Pandas  
- Encode categorical features such as parental education and test preparation  
- Create a final exam score by averaging subject scores  
- Handle missing values using imputation  
- Apply feature scaling for improved model performance  
- Train a Multilinear Regression model  
- Evaluate the model using MSE, RMSE, and R² Score  
- Optimize performance using feature elimination, Ridge, and Lasso regression  

## Key Insights
- Study hours and attendance showed strong positive influence on performance  
- Test preparation course completion improved final exam scores  
- Feature scaling improved coefficient stability  
- Regularization reduced overfitting and improved generalization


## Vehicle Fuel Efficiency Prediction (Polynomial Regression)

This project predicts vehicle fuel efficiency (Miles Per Gallon) based on engine horsepower using Polynomial Regression to capture non-linear relationships.

## Dataset
**Source:** Kaggle – Auto MPG Dataset  
(https://www.kaggle.com/datasets/uciml/autompg-dataset)

## Objectives
- Load and clean the Auto MPG dataset using Pandas  
- Select engine horsepower as the independent variable  
- Handle missing values using mean imputation  
- Apply Polynomial Regression with degrees 2, 3, and 4  
- Scale features to improve model performance  
- Evaluate models using MSE, RMSE, and R² Score  
- Control overfitting using Ridge regularization  

## Key Insights
- MPG shows a non-linear inverse relationship with engine horsepower  
- Higher-degree polynomials improved fit but increased overfitting risk  
- Feature scaling stabilized polynomial model training  
- Ridge regression reduced coefficient magnitude and improved generalization  

