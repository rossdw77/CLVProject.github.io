# CLV Project
Evaluating a variety of modeling techniques to predict customers' lifetime value from a Kaggle dataset (https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr)

## Summary

### What are customers actually worth for an e-commerce retail business?

CLV prediction matters to ensure that marketing teams are catering to their customer population which has varying degrees of value to the business. A business wants to ensure that high value customers are retained and continue to spend, however low and mid value customers make up a larger population and in aggregate drive total revenue. 

The primary target variable for this project is **Total_Amount**

**Recency, Frequency and Spend** are key independent variables to consider, however other features like product category, customer demographics and website engagement were evaluated. Product category is a feature that is **highly linked** to **how much a customer ended up spending**


### What actions can a business take to maximize CLV value?


<img width="1440" height="1920" alt="image" src="https://github.com/user-attachments/assets/3025b16b-2f9e-4ec9-baa1-86402bbe0de0" />



### Design constraints shaped the architecture

The data is traditional of a customer transaction data set as it was heavily right skewed. However, there were key limitations to evaluate:

1. **The total data set was only 1 year and 3 months** which limited how much data the models could be trained on
2. Even after removing outliers, the data has strong variability, **skewing even more heavily to the right than common customer transaction data sets**


<img width="859" height="470" alt="image" src="https://github.com/user-attachments/assets/53decc22-3e79-4533-8b9b-9476627e0f96" />



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

**Test: 9/1/2024 to 3/25/2025**



## Workflow

### End to end coding infrastucture

#### Pipeline


<img width="1440" height="680" alt="image" src="https://github.com/user-attachments/assets/ae5fc767-c4d8-4878-a081-29a3ca629b9b" />



## Modeling Techniques

Expectation is that xgboost would perform the best across both classification and regression due to number of features considered that could help break down right tailed distribution of customers' spend. As the model uses an ensemble method to build sequentially on prior errors, it is better positioned to capture non-linear relationships.


**Hyperparameter Tuning**

Hyperparameter search used randomized cross-validation across both XGBoost and Random Forest models. Given the right skewed spend distribution and relatively limited dataset size, the search grid was deliberately biased toward regularization from the start.
max_depth was capped at 5 and the best result landed at 4, sitting cleanly in the middle of the grid. min_child_weight was kept high at 20 to ensure sufficient observations at each leaf and limit spurious splits on low frequency customers. reg_lambda was set well above the XGBoost default across the entire grid, reflecting a prior expectation that the model would need explicit penalization to generalize across a skewed target. The selected learning rate of 0.01 paired with 500 estimators favored slow, stable convergence over speed.



## Modeling Methodology Summary


<img width="1440" height="840" alt="image" src="https://github.com/user-attachments/assets/4d1e50f5-396c-486c-9a01-ab1c5d1d1aec" />


## Modeling Methodology Detail

1. The starting hypothesis was straightforward: a **Regression** model trained on RFM features and customer attributes should be sufficient to predict CLV. However, it wasn't and lead to poor results (lower train and higher test MAPE). Due to the data limitations described above. Most of the feature importance went into electronics which is helpful to discern high value customers but not as important for lower value as those customers are defined by other factors which will be show in the **Regression by Decile** approach. 


**Results from the Random Forest Model**


<img width="989" height="1014" alt="image" src="https://github.com/user-attachments/assets/403d81d9-712e-4858-8974-77e3a26e1f7c" />


**Shap Values**

<img width="798" height="900" alt="image" src="https://github.com/user-attachments/assets/b6e09b2b-cdbf-4936-83bb-4e9b12133d20" />


**Train**  

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/00ea4a48-067f-4179-b076-711ec0caba21" />



**Test**  

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/ef35b869-2bf6-4502-9f2e-dae0d2e66f2a" />







2. **Classification** approach was the best result but reframes our problem statement solution as we don't know the exact value of the customer but can pinpoint their low, mid, high rating based on key purchasing features. The focus was to have a high recall (low false negatives) as we especially don't want to fail to market to our higher value customer pool. The Xgboost Classifier did a great job with low and high customer rankings, however it was less effective discerning the mid value customers. It was about equal between low and high misaligned predictions. Due to the spend disparity knowing that a customer is high value is helpful but not precise as the range could be anywhere from 1300 TRY to 15000 TRY.



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

[[1194    3  218]  
 [   6 1164  204]  
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




3. The **Multi Staged** approach attempted to combine both: classify first, then regress within each predicted segment. I trained classification models on **true** low, mid and high splits and generated test rating predictions. Train MAPE was acceptable but test MAPE degraded significantly across all bins. The classification step introduced label noise as customers near segment boundaries were misassigned, and the downstream regression models overfit to those mislabeled populations rather than learning the true spend signal.


**Train**


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




4. **Regression by Bins** approach in which a model was trained on each of the 6 bins, resulted in superb performance with test MAPE around 20-25% due to the variation in feature importance across the bins. However, would need to confirm in reality whether it is feasible to maintain multiple models due to data volume, model computation and model maintenance.


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


## Conclusion

CLV prediction is a solved problem in theory and a messy one in practice. This project reflects how I actually work which begins with a clear hypothesis, measure honestly against it, and adapt when the results tell you something the original plan didn't account for.
Flat regression failed not because of poor execution but because a single model cannot adequately capture a customer population that spans 5x the mean with a limited time series. The classification approach reframed the problem usefully but sacrificed the precision that makes CLV actionable for marketing budget allocation. The binned regression architecture resolved both constraints, with separate models per spend segment each calibrated to the variance profile of that population, achieving 20-25% test MAPE across all bins. However, that modeling technique would need to be assessed among Marketing leadership, Data Engineering and the Data Science team as to whether it could be smoothly integrated into production.
The true output of this project isn't a model. It's a repeatable framework for **navigating the gap between what is technically optimal and what is operationally viable. Every modeling decision involves a tradeoff between accuracy and maintainability, or precision and interpretability. Surfacing those tradeoffs clearly is what separates analysis from a business recommendation.**
