# Credit_Risk_Analysis

## Overview 

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. For that reason,in this analysis different techniques were used to train and evaluate models with unbalanced classes. This analysis aims to understand how to utilize `Machine Learning` statistical algorithms to make predictions based on data patterns provided. 

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, the data was oversampled by using the `RandomOverSampler` and `SMOTE` algorithms and undersampled by using the `ClusterCentroids` algorithm. 

The [dataset](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv.zip) from the LendingClub has an unbalanced classification problem because the number of good loans outweighs the number of risky loans. To balance out the classifications to allow for more meaningful predictions and improve the accuracy score, various Machine Learning algorithms were applied to resample the data like `SMOTEENN`, 
`BalancedRandomForestClassifier` and `EasyEnsembleClassifier`.
