# Credit_Risk_Analysis

## Overview 

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. For that reason,in this analysis different techniques were used to train and evaluate models with unbalanced classes. This analysis aims to understand how to utilize `Machine Learning` statistical algorithms to make predictions based on data patterns provided. 

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, the data was oversampled by using the `RandomOverSampler` and `SMOTE` algorithms and undersampled by using the `ClusterCentroids` algorithm. 

The [dataset](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv.zip) from the LendingClub has an unbalanced classification problem because the number of good loans outweighs the number of risky loans. To balance out the classifications to allow for more meaningful predictions and improve the accuracy score, various Machine Learning algorithms were applied to resample the data like `SMOTEENN`, 
`BalancedRandomForestClassifier` and `EasyEnsembleClassifier`.

## Result 

Using the `imbalanced-learn` and `scikit-learn` libraries, three machine learning models were evaluated by using resampling to determine which is better at predicting credit risk. 

First, oversampling **RandomOverSampler** and **SMOTE** algorithms were used; then undersampling **ClusterCentroids** algorithm was used to resample the dataset, view the count of the target classes, and train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

The original dataset contained 115,675 loan applications in Q1 of 2019. To determine whether the application was considered "low" or "high" risk, the **loan status** was used.  Applications that had current as the loan status were classified as **low risk** and the remaining as **high risk**. This reduced the dataset to **68,817** total applications with **99%** classified as "low risk". 

![Image_1](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_1.png)

Using the **75%-25% method** to split the data for training vs. testing, 51,352 "low risk" and 260 "high risk" applications were categorized into the training set. 17,118 "low risk" and 87 "high risk" applications were categorized into the test set.

![Image_2](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_2.png)

### Resampling Models to Predict Credit Risk

#### Oversampling
`The RandomOverSampler` model instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. For this analysis, the oversampling result classified **51,352** records for each "high risk" and "low risk". 

![Image_3](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_3.png)

To assess the accuracy score of the model,use the `balanced_accuracy_score` module was used and the accuracy score was found **65%**. However, because of this number can be misleading, especially in an unbalanced dataset, to assess the results further the `classification_report_imbalanced` module was used.

![Image_4](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_4.png)

The "High Risk" precision rate was only 1% with the recall at 61% giving this model an F1 score of 2%.
"Low Risk" had a precision rate of 100% and recall at 68%.

![Image_5](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_5.png)
