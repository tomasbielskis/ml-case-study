import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/churn.csv')
num_vars = ['avg_dist','avg_rating_by_driver','avg_rating_of_driver','surge_pct','avg_surge','trips_in_first_30_days','weekday_pct']
cat_vars = ['city','phone','luxury_car_user']


df['churn'] = np.where(pd.to_datetime(df['last_trip_date']).dt.month > 5, 0, 1)


df.loc[df['phone']=="iPhone", 'phone'] = 0
df.loc[df['phone']=="Android", 'phone'] = 1

df = pd.concat([df, pd.get_dummies(df['city'])], axis=1).drop('city',axis=1)

df = df.drop(['last_trip_date','signup_date'],axis=1)

df.fillna(-1, inplace=True)

X = df.drop('churn',axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log = LogisticRegression()
mod = log.fit(X_train,y_train)
y_pred = mod.predict_proba(X_test)[:, 1]

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list
    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''

    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()

tpr, fpr, thresholds = roc_curve(y_pred, y_test)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot")
plt.show()
