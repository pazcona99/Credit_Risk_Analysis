# Credit_Risk_Analysis
Employing supervised machine learning to evaluate credit card risk and credit assessment from an established lending dataset.

# Purpose

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, there is a need to employ different techniques to train and evaluate models with unbalanced classes. This code uses the `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we oversample the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then, we use a combinatorial approach of over- and undersampling using the `SMOTEENN` algorithm. Next, we compare two new machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. These methods will help evaluate the performance of these models so that a recommendation on whether they can be used to predict credit risk can be made.

# Results

- Oversample / RandomOverSampler
    - ![Oversampling_RandomOverSampler](./Resources/Oversampling_RandomOverSampler.PNG)

    - Balanced accuracy scores = 0.64
    - High_risk Precision = 0.01
    - High_risk recall (Sensitivity) = 0.64
    - High_risk F1 score = 0.02
    - Low_risk Precision = 1.0
    - Low_risk recall = 0.66
    - Low_risk F1 score = 0.79
    - The high_risk precision score doesn't looks good and this make the f1 score result in worse

    
- Oversample / SMOTE
    - ![Oversampling_SMOTE](./Resources/Oversampling_SMOTE.PNG)

    - Balanced accuracy scores = 0.63
    - High_risk Precision = 0.01
    - High_risk recall (Sensitivity) = 0.62
    - High_risk F1 score = 0.02
    - Low_risk Precision = 1.0
    - Low_risk recall = 0.64
    - Low_risk F1 score = 0.78
    - The high_risk precision score doesn't looks good and this make the f1 score result in worse

- Undersample / ClusterCentroids
    - ![Undersampling_ClusterCentroids](./Resources/Undersampling_ClusterCentroids.PNG)

    - Balanced accuracy scores = 0.51
    - High_risk Precision = 0.01
    - High_risk recall (Sensitivity) = 0.57
    - High_risk F1 score = 0.01
    - Low_risk Precision = 1.0
    - Low_risk recall = 0.45
    - Low_risk F1 score = 0.62
    - Undersampleing with ClusterCentroids reports are even worse than the oversample models above. 

- Combine (Over- and Undersample) / SMOTEENN algorithm
    - ![Combination_Sampling_SMOTEEN](./Resources/Combination_Sampling_SMOTEEN.PNG)

    - Balanced accuracy scores = 0.64
    - High_risk Precision = 0.01
    - High_risk recall (Sensitivity) = 0.74
    - High_risk F1 score = 0.02
    - Low_risk Precision = 1.0
    - Low_risk recall = 0.55
    - Low_risk F1 score = 1.0
    - We don't see much improvement using resampling with SMOTEENN, only some of the metrics such as recall score has an improvement over undersampling.

- BalanceRandomForestClassifier
    - ![BalancedRandomForestClassifier](./Resources/BalancedRandomForestClassifier.PNG)

    - Balanced accuracy scores = 0.999
    - High_risk Precision = 0.83
    - High_risk recall (Sensitivity) = 1.0
    - High_risk F1 score = 0.91
    - Low_risk Precision = 1.0
    - Low_risk recall = 1.0
    - Low_risk F1 score = 1.0
    - Using ensemble algorithms with Balance Random forest Classifier is impressive, we have a great balanced accuracy score, all the metrics on classification report looks great.

- EasyEnsebleClassifier
    - ![EasyEnsembleClassifier](./Resources/EasyEnsembleClassifier.PNG)

    - Balanced accuracy scores = 1.0
    - High_risk Precision = 0.92
    - High_risk recall (Sensitivity) = 0.94
    - High_Risk_F1 score = 0.16
    - Low_Risk_F1 score = 0.97
    - Low_risk Precision = 1.0
    - Low_risk recall = 0.94
    - Low_risk F1 score = 0.92
    - Using ensemble algorithms with Easy Enseble Classifier is impressive, we have a great balanced accuracy score, all the metrics on classification report are score in 1.0

# Summary

## Recomendation