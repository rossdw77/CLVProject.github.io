# CLVProject
Evaluating a variety of modeling techniques to predict customers' lifetime value from a Kaggle dataset (https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr)

## Summary

### What are customers actually worth for an e-commerce retail business?

CLV prediction matters to ensure that marketing teams are catering to their customer population which has varying degrees of value to the business. Of course, the business wants to ensure that high value customers are retained and continue to spend, however low and mid value customers make up a larger population and in aggregate drive total revenue. 

The primary target variable for this project is **Total_Amount**

**Recency, Frequency and Spend ** are key independent variables to consider, however other features like product category, customer demographics and website engagement were evaluated

## Data

(Summary of data set - link to kaggle section)


### Design constraints shaped the architecture

The data is traditional of a customer transaction data set as it was heavily right skewed. However, there were key limitations to evaluate:

1. **The total data set was just over 1 year** which limited how much data the models could be trained on
2. Even after removing outliers, the data has strong variability, **skewing even more heavily to the right than common customer transaction data sets**

(add min max dates and viz for key dists)

<img width="859" height="470" alt="image" src="https://github.com/user-attachments/assets/3281b451-781d-4a74-ad43-8a9712cb7085" />




## Workflow

### End to end coding infrastucture

#### Pipeline

1. Config
   Desc: Key parameters to consider in model processing: column identifiers, train/test date split, spend binning

2. data_loader
   
   ├── Raw transaction data
   └── Outlier Thresholds

3. Visualizations
   ├── Spend distribution analysis
   ├── Categorical slices
   └── Time series review

4. Preprocessing
   - Customer aggregation for modeling
   - RFM computation (Recency, Frequency, Monetary)

5. model_prep
   ├─ Select features
   ├── Spend binning
   └── CLV ranking
   - Regression and classification train/test splits

6. classification_models
   ├── Logistic Regression
   ├── Xgboost classifer
   └── Decision Trees

7. regression_models
   ├── Linear Regression
   ├── Xgboost regressor
   └── Decision Trees

9. statistical_models
   ├── BTYD - NBD
   ├── Gamma Gamma

11. Evaluation
   ├── Regression evaluators: Train/test MAPE
   └── Classification evaluators: ROC/AUC, classification report
       SHAP TreeExplainer


## Modeling Methodology

1. Regression based approach lead to poor results (higher train and low test MAPE) Due to the data limitations described above.
2. Classification approach was the best result but reframes our problem statement solution as we don't know the exact value of the customer but can pinpoint their low, mid, high rating based on key features (talk about precision, recall - more important to have high recall as we don't want to disregard marketing to a high value customer)
3. Multi Staged approach in which I trained classification models on true low, mid and high splits and generated test rating predictions. Then trained regression models on each of the buckets and passed in test data hoping for better regression MAPE accuracy. However, this was no very accurate with high MAPE
4. Regression by decile approach in which a model was trained on each of the deciles resulted in great performance with MAPE around 20-25%. However, not very realistic to maintain 10 models in reality due to data volume, model computation and overall complexity (i.e. many features)


## Modeling Techniques

Expectation is that xgboost would perform the best across both classification and regression due to number of features considered that could help break down right tailed distribution of customers' spend. As model is trained off the residuals to minimize feature split error.


## Statistical Modeling


## Model Summary

- snapshots of AUC/ROC and MAPE

### Regression Models


### Classification Models

- mid, low and high confusion matricies

### Classification to Regression Models

- work on adding the function for the MAPE rerouting and how the misspecified classification lead to the worst scores


### Regression by decile Models


## Conclusion

- The CLV project helped me put in perspective what it takes to build an end to end machine learning process
- My original hypothesized plan did not go fully as expected and I had to deviate to adjust to find the most optimal accuracy across regression and classification models
- I not only learned which models to train but how to put the outputs in perspective to the business (how we can leverage the reg models for top deciles - explain performance and what MAPE actually means in this case)
