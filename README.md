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

#### - Oversampling
`The RandomOverSampler` model instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. For this analysis, the oversampling result classified **51,352** records for each "high risk" and "low risk". 

![Image_3](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_3.png)

To assess the accuracy score of the model,use the `balanced_accuracy_score` module was used and the accuracy score was found **65%** (0.645). However, because of this number can be misleading, especially in an unbalanced dataset, to assess the further results, the `classification_report_imbalanced` module was used.

![Image_4](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_4.png)

The **"High Risk"** precision rate was only 1% with the recall at 61% giving this model an F1 score of 2%.
**"Low Risk"** had a precision rate of 100% and recall at 68%.

![Image_5](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_5.png)

#### - SMOTE 
Synthetic Minority Oversampling Technique (SMOTE) model, like `RandomOverSampler` increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection.

With `SMOTE` model, balanced accuracy score discreased to **62%** (0.623). 

![Image_6](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_6.png)

Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 61% giving this model an F1 score of 2%. "Low Risk" had a precision rate of 100% and recall at 64%.

![Image_7](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_7.png)

#### - Undersampling
Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.

`ClusterCentroids` model, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified **260** records each as High Risk and Low Risk.

![Image_8](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_8.png)

Balanced accuracy score was lower than the oversampling models at **53%** (0.529).

![Image_9](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_9.png)

The **"High Risk"** precision rate again was only at 1% with the recall at 61% giving this model an F1 score of 1%.
**"Low Risk"** had a precision rate of 100% and with a lower recall at 45% compared to the oversampling models.

![Image_10](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_10.png)

 ### SMOTEENN algorithm to Predict Credit Risk
 #### - Combination Sampling
 SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process:
1. Oversample the minority class with SMOTE.
2. Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.
The model classified **68,458** records as "High Risk" and **62,022** as "Low Risk".

![Image_11](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_11.png)

The balanced accuracy score improved to **65%** (0.653) when using a combined sampling model.
![Image_12](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_12.png)

The **"High Risk"** precision rate did not improve was only 1%, however the recall increased to 69% giving this model an F1 score of 2%.**"Low Risk"** still showed a precision rate of 100% with the recall at 62%.
![Image_13](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_13.png)

### Ensemble Classifiers to Predict Credit Risk
The two ensemble algorithms, `Balanced Random Forest Classifier` and `Easy Ensemble AdaBoost Classifier`, were compared to determine which algorithm results in the best performance and that reduce bias to predict credit risk.
#### - BalancedRandomForestClassifier
`BalancedRandomForestClassifier` model, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.

The balanced accuracy score increased to **78.8%** for this model.

![Image_14](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_14.png)

The **"High Risk** precision rate increased to 4% with the recall at 67% giving this model an F1 score of 7%.
**"Low Risk"** still had a precision rate of 100% with the recall at 91%.
The top feature by importance was **"total_rec_prncp"** at 7.4% of the total.

![Image_15](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_15.png)

#### - EasyEnsembleClassifier
`EasyEnsembleClassifier` model, a set of classifiers where individual decisions are combined to classify new examples.
The balanced accuracy score increased to **92.5%** with this model.

![Image_16](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_16.png)

The **"High Risk** precision rate increased to 7% with the recall at 91% giving this model an F1 score of 14%.
**"Low Risk"** still had a precision rate of 100% with the recall now at 94%.

![Image_17.png](https://github.com/duygusimsek/Credit_Risk_Analysis/blob/main/Images/Image_17.png)

## Summary
Working with balanced accuracy, the highest compared accuracy between 0 and 1 and is closest to 1 is the best machine learning model. All the models were used to perform the credit risk analysis show weak precision in determining if credit risk is high. Reviewing the six models, the **`EasyEnsembleClassifer`** model displayed the best results with an **accuracy rate of 93.2%** and a **7% precision rate** when predicting “high-risk“ candidates. When comparing the other models, the **sensitivity rate (recall)** was also the highest at **94%**, which shows it detects almost all high-risk credit. And the **balanced accuracy** for the **`EasyEnsembleClassifer`** model is at **92.5%**, the other models were below 80%.

Also, when considering the original dataset had 99% of the applications classified as "Low Risk" with only 1% of the data classified in the "High Risk" category. This may skew the results as there is a risk that the Machine Learning algorithms are creating clusters drawing from too small of a dataset of actual "High Risk" applications. Low precision shows a lot of low-risk credits are still falsely detected as high risk which would penalize the bank's credit strategy and infer its revenue by missing those business opportunities. 
