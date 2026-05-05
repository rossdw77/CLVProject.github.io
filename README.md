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

**Train: 1/1/2023 to 8/31/2024**

**Test: 9/1/2024 to 3/25/2024**



## Workflow

### End to end coding infrastucture

#### Pipeline


| Module | Description |
|---|---|
| `config` | Column identifiers, train/test date split, spend binning parameters |
| `data_loader` | Raw transaction ingestion and outlier threshold definitions |
| `visualizations` | Spend distribution, categorical slices, time series review |
| `preprocessing` | Customer aggregation, RFM feature computation |
| `model_prep` | Feature selection, spend binning, CLV ranking, train/test splits |
| `classification_models` | Logistic Regression, XGBoost Classifier, Decision Trees |
| `regression_models` | Linear Regression, XGBoost Regressor, Decision Trees |
| `statistical_models` | BTYD/NBD, Gamma-Gamma |
| `evaluation` | MAPE (regression), ROC/AUC + classification report, SHAP TreeExplainer |



## Modeling Techniques

Expectation is that xgboost would perform the best across both classification and regression due to number of features considered that could help break down right tailed distribution of customers' spend. As the model uses an ensemble method to build sequentially on prior errors, it is better positioned to capture non-linear relationships.


**Hyperparameter Tuning**

I used CV randomized search for xgboost and random forest models in order to find the best parameters to train models and avoid overfitting.

Due to the computational power needed to process cross validation plus each parameters' grid, I began by choosing a range of values that were on the lower side to see if the best fit parameters stayed on the lower end of the range or continually were at the higher end of the range. I was especially cognizant of the max_depth and wanted to keep that lower in order to avoid overfitting. However, I did keep the min_child_weight on the higher side in order to ensure I had enough data points so that there were not too many tree splits to keep things more simplistic. My reg lambda score was also higher in order to ensure there was no overfitting. 


[Describe your tuning approach. Key decisions to mention: max_depth grid, min_child_weight bounds, reg_lambda behavior, early stopping implementation.]


## Modeling Methodology Summary

1. **Regression** based approach lead to poor results. (lower train and higher test MAPE) Due to the data limitations described above. Most of the feature importance went into electronics which is helpful to discren high value customers but not as important for lower value as those customers are defined by other factors which will be show in the **Regression by Decile** approach.


**Results from the Random Forest Model**


<img width="989" height="1014" alt="image" src="https://github.com/user-attachments/assets/403d81d9-712e-4858-8974-77e3a26e1f7c" />



<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/00ea4a48-067f-4179-b076-711ec0caba21" />


**Train**


<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/ef35b869-2bf6-4502-9f2e-dae0d2e66f2a" />




**Test**

<img width="798" height="900" alt="image" src="https://github.com/user-attachments/assets/b6e09b2b-cdbf-4936-83bb-4e9b12133d20" />



2. **Classification** approach was the best result but reframes our problem statement solution as we don't know the exact value of the customer but can pinpoint their low, mid, high rating based on key purchasing features. The focus was to have a high recall (low false negatives) as we especially don't want to fail to market to our predicted higher value customer pool. The Xgboost Classifier did a great job with low and high customer rankings, however it was less effective discerning the mid value customers. It was about equal between low and high misaligned predictions. 


**Results from Xgboost Classifier**


[train] ROC AUC: 0.9403

[train] Classification Report:
              precision    recall  f1-score   support

        high       0.85      0.84      0.85      1415
         low       0.87      0.85      0.86      1374
         mid       0.70      0.72      0.71      1373

    accuracy                           0.80      4162
   macro avg       0.81      0.80      0.80      4162
weighted avg       0.81      0.80      0.81      4162

[train] Confusion Matrix:

[[1194    3  218]/n
 [   6 1164  204]/n
 [ 204  178  991]]




[test] ROC AUC: 0.9075

[test] Classification Report:
              precision    recall  f1-score   support

        high       0.80      0.77      0.78      1268
         low       0.86      0.83      0.84      1443
         mid       0.61      0.66      0.63      1250

    accuracy                           0.76      3961
   macro avg       0.76      0.75      0.75      3961
weighted avg       0.76      0.76      0.76      3961


[test] Confusion Matrix:
[[ 975    1  292]
 [   6 1193  244]
 [ 236  191  823]]






 
3. **Multi Staged** approach in which I trained classification models on **true** low, mid and high splits and generated test rating predictions. Then trained regression models on each of the buckets and passed in test data hoping for better MAPE scores compared to the **Regression** approach. However, this was not a very promising solution that ended resulted in high MAPE.


**Train**


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
      <th>MAE</th>
      <th>MAPE</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Low</th>
      <td>98.8730</td>
      <td>0.5767</td>
    </tr>
    <tr>
      <th>Mid</th>
      <td>309.4247</td>
      <td>0.2604</td>
    </tr>
    <tr>
      <th>High</th>
      <td>1779.9568</td>
      <td>0.3370</td>
    </tr>
  </tbody>
</table>
</div>




**Test** 



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
      <th>MAE</th>
      <th>MAPE</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Low</th>
      <td>173.2437</td>
      <td>0.6757</td>
    </tr>
    <tr>
      <th>Mid</th>
      <td>848.2739</td>
      <td>0.6582</td>
    </tr>
    <tr>
      <th>High</th>
      <td>2784.5553</td>
      <td>0.9092</td>
    </tr>
  </tbody>
</table>
</div>






4. **Regression by Bins** approach in which a model was trained on 6 bins, which resulted in great performance with test MAPE around 20-25%. However, not very realistic to maintain 10 models in reality due to data volume, model computation and model maintenance.


**6 Bins**

Categories (6, interval[float64, right]): [(36.066, 159.81] < (159.81, 439.852] < (439.852, 1299.55] < (1299.55, 3441.978] < (3441.978, 7298.629] < (7298.629, 15077.561]]


<img width="1809" height="1790" alt="image" src="https://github.com/user-attachments/assets/f9008ba8-1259-47aa-abff-ff7ae44c8cf0" />


**Train**


MAE:  $21.17
MAPE: 29.11%
MAE:  $48.47
MAPE: 18.48%
MAE:  $139.95
MAPE: 19.19%
MAE:  $348.15
MAPE: 17.05%
MAE:  $670.95
MAPE: 13.77%
MAE:  $1,411.49
MAPE: 13.74%


**Test**


MAE:  $27.30
MAPE: 29.69%
MAE:  $68.99
MAPE: 27.20%
MAE:  $195.40
MAPE: 27.08%
MAE:  $479.22
MAPE: 22.72%
MAE:  $907.31
MAPE: 19.89%
MAE:  $2,019.83
MAPE: 19.12%




## Statistical Modeling

TBD

## Conclusion

- The CLV project helped me put in perspective what it takes to build an end to end machine learning process. You cannot just be good at one aspect. There must be a coordinated effort to balance business objectives, coding, model building, and model deployment / execution.
- My original hypothesized plan did not go fully as expected and I had to deviate to adjust to find the most optimal accuracy by testing regression and classification methodologies.
- I not only learned how to develop models and best practices in my evolving framework, but focused on putting the outputs in perspective to the business. For instance, MAPE could at least be leveraged to predict spend amounts for the higher deciles which are essential to driving sustaining and growing the business.
