# Telecom Customer Churn Analysis

## Project Overview
This project analyzes customer churn in a telecommunications company using machine learning models. The analysis focuses on identifying key factors that contribute to customer churn and building predictive models to detect potential churners.

## Dataset Description
- Dataset contains ~100K rows and 230 columns
- Features include call details, recharge patterns, and usage metrics
- Data spans 3 months (months 6-8) with month 9 used for churn labeling
- High-value customers (70th percentile by recharge amount) are analyzed

## Key Features Analyzed
- Recharge patterns and amounts
- Voice usage (Local, STD, ISD)
- Data usage (2G/3G)
- Roaming behavior
- Special call rates usage
- Average Revenue Per User (ARPU)

## Methodology

### Data Preprocessing
1. Missing value treatment
2. Feature engineering
   - Recharge behavior metrics
   - Usage pattern indicators
   - Service type utilization
3. Standardization of numerical variables
4. SMOTE for handling class imbalance

### Models Developed
1. Logistic Regression with RFE
   - Interpretable model with key feature selection
   - ROC-AUC: 0.93
   - Test Sensitivity: 75.9%

2. PCA + Logistic Regression
   - Dimension reduction to 105 components
   - Improved model generalization

3. PCA + Random Forest
   - Non-linear relationships capture
   - Better handling of complex patterns

## Key Findings

### Churn Indicators
1. **Roaming Usage**: Higher churn probability for roaming customers
2. **Usage Patterns**: 
   - Lower incoming calls in action phase
   - Reduced local calls to mobile operators
3. **Recharge Behavior**: 
   - Lower recharges in action period
   - Higher recharges in good phase
4. **Service Impact**:
   - STD & ISD usage correlates with higher churn
   - Special services require optimization

### Recommendations
1. Optimize roaming plans and pricing
2. Focus on maintaining consistent usage patterns
3. Monitor recharge behavior changes
4. Review and improve STD/ISD services
5. Enhance special service offerings

## Technical Requirements
```python
Required Libraries:
- pandas
- numpy
- sklearn
- statsmodels
- seaborn
- matplotlib
- imblearn
```

## Usage
1. Load and preprocess data
2. Run feature engineering steps
3. Train models using the provided notebooks
4. Use trained models for churn prediction

## Model Performance
- Best performing model: PCA + Logistic Regression
- Metrics:
  - ROC-AUC: 0.93
  - Sensitivity: 75.9%
  - Specificity: 71.2%
  - F1-Score: 0.73

## Future Improvements
1. Incorporate more recent data
2. Add customer demographic information
3. Develop real-time churn prediction system
4. Implement model monitoring system
