# loan_default_prediction

I conducted loan default prediction by applying six machine learning algorithms (Logistic regression, Ridge, LASSO, Gradient Boosting, SVM, Random Forest) on individual level loan data from Lending Club.  The prediction model I developed will help Lending Club to detect whether the new borrowers will be default (do not pay back the loan in time) so that Lending Club can avoid the risk to losing money.

I cleaned the data, constructed outcome metric and features, and selected features.

The data had a serious problem of unbalanced classes: a relatively small proportion of "Default" cases vs. large proportion of "Non-Default" cases, which led to learning algorithms being less capable or even unable to predict "Default." I employed two methods to solve this problem. The first one is undersampling majority classes and/or oversampling minority classes before training. The second one is changing the prediction threshold based on ROC curve.

For each algorithm, I did ten fold cross validation to calculate confusion matrices. For each confusion matrix from each fold cross validation, I calculate performance statistics for both "Default" class and "Non-Default" class: Precision, Recall, and F-score, and the total accuracy rate. I compared the six machine learning algorithms based on the average of performance statistics across ten fold cross validation. Random Forest outperformed other algorithms and had all the performance statistics being over 0.99. All the work was done using R.
