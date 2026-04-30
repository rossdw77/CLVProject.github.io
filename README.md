# CLVProject
Evaluating a variety of modeling techniques to predict customers' lifetime value from a Kaggle dataset (https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr)

## Summary

### What are customers actually worth for an e-commerce retail business?

CLV prediction matters to ensure that marketing teams are catering to their customer population which has varying degrees of value to the business. Of course, the business wants to ensure that high value customers are retained and continue to spend, however low and mid value customers make up a larger population and in aggregate drive total revenue. 

The primary target variable for this project is **Total_Amount**

**Recency, Frequency and Spend** are key independent variables to consider, however other features like product category, customer demographics and website engagement were evaluated. Product category is a feature that is **highly linked** to **how much a customer ended up spending**

## Data

Order_ID
Unique identifier for each transaction. Format: ORD_XXXXXX (6-digit number)

Customer_ID
Unique identifier for each customer. Format: CUST_XXXXX (5-digit number)

Date
Transaction date when the order was placed. Range: 2023-01-01 to 2024-03-26

Age
Customer's age in years. Range: 18-75 years old

Gender
Customer's gender. Values: Male, Female, Other

City
Customer's city location in Turkey. 10 major cities included

Product_Category
Category of purchased product. 8 categories: Electronics, Fashion, Home & Garden, Sports, Books, Beauty, Toys, Food

Unit_Price
Price per unit of the product in Turkish Lira (TRY). Varies by category

Quantity
Number of units purchased in the transaction. Range: 1-5 units

Discount_Amount
Total discount applied to the order in TRY. Zero if no discount applied

Total_Amount
Final amount paid after discount (Unit_Price × Quantity - Discount_Amount)

Payment_Method
Method used for payment. Options: Credit Card, Debit Card, Digital Wallet, Bank Transfer, Cash on Delivery

Device_Type
Device used to make the purchase. Options: Mobile, Desktop, Tablet

Session_Duration_Minutes
Time spent on website during the session in minutes. Range: 1-120 minutes

Pages_Viewed
Number of pages viewed during the shopping session. Range: 1-50 pages

Is_Returning_Customer
Whether the customer has made previous purchases. Values: True (returning) or False (new customer)

Delivery_Time_Days
Number of days taken to deliver the order. Range: 1-30 days

Customer_Rating
Customer satisfaction rating for the order. Scale: 1-5 stars (1=very dissatisfied, 5=very satisfied)


### Design constraints shaped the architecture

The data is traditional of a customer transaction data set as it was heavily right skewed. However, there were key limitations to evaluate:

1. **The total data set was only 1 year and 3 months** which limited how much data the models could be trained on
2. Even after removing outliers, the data has strong variability, **skewing even more heavily to the right than common customer transaction data sets**


<img width="859" height="470" alt="image" src="https://github.com/user-attachments/assets/3281b451-781d-4a74-ad43-8a9712cb7085" />

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Recency</th>
      <td>4162.0</td>
      <td>293.969966</td>
      <td>65.509280</td>
      <td>207.0000</td>
      <td>237.0000</td>
      <td>280.00</td>
      <td>341.0000</td>
      <td>449.0000</td>
    </tr>
    <tr>
      <th>duration</th>
      <td>4162.0</td>
      <td>66.494954</td>
      <td>72.858306</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>41.00</td>
      <td>128.0000</td>
      <td>238.0000</td>
    </tr>
    <tr>
      <th>Frequency</th>
      <td>4162.0</td>
      <td>2.182364</td>
      <td>1.363212</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>2.00</td>
      <td>3.0000</td>
      <td>9.0000</td>
    </tr>
    <tr>
      <th>Quantity</th>
      <td>4162.0</td>
      <td>6.602114</td>
      <td>4.680904</td>
      <td>1.0000</td>
      <td>3.0000</td>
      <td>5.00</td>
      <td>9.0000</td>
      <td>32.0000</td>
    </tr>
    <tr>
      <th>Total_Amount</th>
      <td>4162.0</td>
      <td>2652.033013</td>
      <td>3307.937031</td>
      <td>37.0663</td>
      <td>439.8525</td>
      <td>1299.55</td>
      <td>3441.9775</td>
      <td>15076.5614</td>
    </tr>
  </tbody>
</table>
</div>



Ultimately the data was split using the dates below:

Train: 1/1/2023 to 8/31/2024
Test: 9/1/2024 to 3/25/2024



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


## Modeling Methodology Summary

1. **Regression** based approach lead to poor results. (lower train and higher test MAPE) Due to the data limitations described above. Most of the feature importance went into electronics which is helpful to discren high value customers but not as important for lower value as those customers are defined by other factors which will be show in the **Regression by Decile** approach.


**Results from the Random Forest Model**


<img width="989" height="1014" alt="image" src="https://github.com/user-attachments/assets/403d81d9-712e-4858-8974-77e3a26e1f7c" />



<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/00ea4a48-067f-4179-b076-711ec0caba21" />



<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/ef35b869-2bf6-4502-9f2e-dae0d2e66f2a" />



<img width="798" height="900" alt="image" src="https://github.com/user-attachments/assets/b6e09b2b-cdbf-4936-83bb-4e9b12133d20" />



2. **Classification** approach was the best result but reframes our problem statement solution as we don't know the exact value of the customer but can pinpoint their low, mid, high rating based on key purchasing features. The focus was to have a high recall (low false negatives) as we don't want to fail to send marketing to our predicted higher value customer pool.







 
3. Multi Staged approach in which I trained classification models on true low, mid and high splits and generated test rating predictions. Then trained regression models on each of the buckets and passed in test data hoping for better regression MAPE accuracy. However, this was not very accurate with high MAPE
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
