# CLV Project
Evaluating a variety of modeling techniques to predict customers' lifetime value from a Kaggle dataset (https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr)

## Summary

### What are customers actually worth for an e-commerce retail business?

CLV prediction matters to ensure that marketing teams are catering to their customer population which has varying degrees of value to the business. A business wants to ensure that high value customers are retained and continue to spend, however low and mid value customers make up a larger population and in aggregate drive total revenue. 

The primary target variable for this project is **Total_Amount**

**Recency, Frequency and Spend** are key independent variables to consider, however other features like product category, customer demographics and website engagement were evaluated. Product category is a feature that is **highly linked** to **how much a customer ended up spending**


### What actions can a business take to maximize CLV value?


<img width="1440" height="1796" alt="image" src="https://github.com/user-attachments/assets/f357eda9-170b-45c4-a6a5-4171945ab64d" />




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


<img width="1440" height="680" alt="image" src="https://github.com/user-attachments/assets/fabd5da5-e09d-4978-8e82-edc3a08e00a0" />



## Modeling Techniques

Expectation is that xgboost would perform the best across both classification and regression due to number of features considered that could help break down right tailed distribution of customers' spend. As the model uses an ensemble method to build sequentially on prior errors, it is better positioned to capture non-linear relationships.


**Hyperparameter Tuning**

Hyperparameter search used randomized cross-validation across both XGBoost and Random Forest models. Given the right skewed spend distribution and relatively limited dataset size, the search grid was deliberately biased toward regularization from the start.
max_depth was capped at 5 and the best result landed at 4, sitting cleanly in the middle of the grid. min_child_weight was kept high at 20 to ensure sufficient observations at each leaf and limit spurious splits on low frequency customers. reg_lambda was set well above the XGBoost default across the entire grid, reflecting a prior expectation that the model would need explicit penalization to generalize across a skewed target. The selected learning rate of 0.01 paired with 500 estimators favored slow, stable convergence over speed.



## Modeling Methodology Summary


<img width="1440" height="840" alt="image" src="https://github.com/user-attachments/assets/81ffa788-7f78-4669-be69-9050af7de999" />

<br>
<br>



## Modeling Methodology Detail

1. The starting hypothesis was straightforward: a **Regression** model trained on RFM features and customer attributes should be sufficient to predict CLV. However, it wasn't and lead to poor results (lower train and higher test MAPE). Due to the data limitations described above. Most of the feature importance went into electronics which is helpful to discern high value customers but not as important for lower value as those customers are defined by other factors which will be show in the **Regression by Decile** approach. 


**Results from the Random Forest Model**


<img width="989" height="1014" alt="image" src="https://github.com/user-attachments/assets/403d81d9-712e-4858-8974-77e3a26e1f7c" />


**Shap Values**

<img width="798" height="900" alt="image" src="https://github.com/user-attachments/assets/b6e09b2b-cdbf-4936-83bb-4e9b12133d20" />


**Train vs Test MAPE**  

<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/60feb00b-7a8e-40dd-be28-860dc400b419" />



<br>
<br>

2. **Classification** approach was the best result but reframes our problem statement solution as we don't know the exact value of the customer but can pinpoint their low, mid, high rating based on key purchasing features. The focus was to have a high recall (low false negatives) as we especially don't want to fail to market to our higher value customer pool. The Xgboost Classifier did a great job with low and high customer rankings, however it was less effective discerning the mid value customers. It was about equal between low and high misaligned predictions. Due to the spend disparity knowing that a customer is high value is helpful but not precise as the range could be anywhere from 1300 TRY to 15000 TRY.



**Results from Xgboost Classifier**


Train ROC AUC: 0.9403

<div>
<img width="278" height="115" alt="image" src="https://github.com/user-attachments/assets/4f7c3d44-72ba-4fbb-a157-5988f741b9b8"/> 

<img width="200" height="144" alt="image" src="https://github.com/user-attachments/assets/ff07c582-2ce8-4670-bc18-8a4433fd3123" />
 </div>

<br>
<br>

Test ROC AUC: 0.9075

<div>
<img width="278" height="116" alt="image" src="https://github.com/user-attachments/assets/b0f64bf0-ffcf-4563-9215-adb68580d038"/> 

<img width="200" height="140" alt="image" src="https://github.com/user-attachments/assets/0366d850-8dcf-4038-b785-fc8b0b213de8" />
 </div>
 

<br>
<br>


3. The **Multi Staged** approach attempted to combine both: classify first, then regress within each predicted segment. I trained classification models on **true** low, mid and high splits and generated test rating predictions. Train MAPE was acceptable but test MAPE degraded significantly across all bins. The classification step introduced label noise as customers near segment boundaries were misassigned, and the downstream regression models overfit to those mislabeled populations rather than learning the true spend signal.


**Train vs Test**


<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/af49de38-78d2-4673-ab90-77157296a2a2" />


<br>
<br>


4. **Regression by Bins** approach in which a model was trained on each of the 6 bins, resulted in superb performance with test MAPE around 20-25% due to the variation in feature importance across the bins. However, would need to confirm in reality whether it is feasible to maintain multiple models due to data volume, model computation and model maintenance.


**6 Bins**

Categories (6, interval[float64, right]): [(36.066, 159.81] < (159.81, 439.852] < (439.852, 1299.55] < (1299.55, 3441.978] < (3441.978, 7298.629] < (7298.629, 15077.561]]


<img width="1809" height="1790" alt="image" src="https://github.com/user-attachments/assets/f9008ba8-1259-47aa-abff-ff7ae44c8cf0" />

<br>
<br>

**Train vs Test**


<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/a4e95f95-6170-4d90-8f40-dcd7908c7360" />


<br>
<br>

## Conclusion

CLV prediction is a solved problem in theory and a messy one in practice. This project reflects how I actually work which begins with a clear hypothesis, measure honestly against it, and adapt when the results tell you something the original plan didn't account for.
Flat regression failed not because of poor execution but because a single model cannot adequately capture a customer population that spans 5x the mean with a limited time series. The classification approach reframed the problem usefully but sacrificed the precision that makes CLV actionable for marketing budget allocation. The binned regression architecture resolved both constraints, with separate models per spend segment each calibrated to the variance profile of that population, achieving 20-27% test MAPE across all bins. However, that modeling technique would need to be assessed among Marketing leadership, Data Engineering and the Data Science team as to whether it could be smoothly integrated into production.
The true output of this project isn't a model. It's a repeatable framework for **navigating the gap between what is technically optimal and what is operationally viable. Every modeling decision involves a tradeoff between accuracy and maintainability, or precision and interpretability. Surfacing those tradeoffs clearly is what separates analysis from a business recommendation.**
