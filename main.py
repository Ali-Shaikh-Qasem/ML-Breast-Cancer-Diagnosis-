import pandas as pd  # M = malignant (خبيث), B = benign (حميد)
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

def encode_output_label(df, output_feature: str):
    # encode the output label to 0 for M and 11 for B
    df[output_feature] = df[output_feature].map({'M': 1, 'B': 0})


def minMax_normalization(df):
    # normalize all features to the range 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    df_normalized = pd.DataFrame(df_scaled, columns=df.columns)
    return df_normalized

def split_dataSet(df, output_feature: str):
    X = df.drop(columns=[output_feature])
    y = df[output_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Part 1: K-Nearest Neighbors (KNN)

def euclidean_distance(x1, x2):
    return sum((x1 - x2) ** 2) ** 0.5


def manhattan_distance(x1, x2):
    return sum(abs(x1 - x2))


def cosine_distance(x1, x2):
    return 1 - (sum(x1 * x2) / ((sum(x1 ** 2) ** 0.5) * (sum(x2 ** 2) ** 0.5)))


def accuracy_fun(y_test, y_pred):
    TP = sum((y_pred == 1) & (y_test == 1))
    TN = sum((y_pred == 0) & (y_test == 0))
    return (TP + TN) / len(y_test)


def precision_fun(y_test, y_pred):
    TP = sum((y_pred == 1) & (y_test == 1))
    FP = sum((y_pred == 1) & (y_test == 0))
    return TP / (TP + FP)


def Recall_fun(y_test, y_pred):
    TP = sum((y_pred == 1) & (y_test == 1))
    FN = sum((y_pred == 0) & (y_test == 1))
    return TP / (TP + FN)

def F1_Score_fun(y_test, y_pred):
    precision = precision_fun(y_test, y_pred)
    recall = Recall_fun(y_test, y_pred)
    return 2 * (precision * recall) / (precision + recall)

def knn(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # Accuracy
    accuracy = accuracy_fun(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    # Precision
    precision = precision_fun(y_test, y_pred)
    print(f'Precision: {precision}')
    # Recall
    recall = Recall_fun(y_test, y_pred)
    print(f'Recall: {recall}')
    # F1-Score
    f1_score = F1_Score_fun(y_test, y_pred)
    print(f'F1-Score: {f1_score}')
    # ROC - AUC
    auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", auc)
    return y_pred

def cross_validation_k (X_train, y_train, k_lower_limit, k_upper_limit):
    k_values = range(k_lower_limit, k_upper_limit + 1)
    cv_scores = []

    # Perform cross-validation for each k
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')  # cv = 5 means 5 fold cross validation
        cv_scores.append(np.mean(scores))

    best_k = k_values[np.argmax(cv_scores)]
    return best_k


def roc_plot(y_test, y_pred):
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

# Part 2: Logistic Regression

def logistic_regression(X_train, X_test, y_train, y_test):

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Accuracy
    accuracy = accuracy_fun(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    # Precision
    precision = precision_fun(y_test, y_pred)
    print(f'Precision: {precision}')
    # Recall
    recall = Recall_fun(y_test, y_pred)
    print(f'Recall: {recall}')
    # F1-Score
    f1_score = F1_Score_fun(y_test, y_pred)
    print(f'F1-Score: {f1_score}')
    # ROC - AUC
    auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", auc)
    return y_pred

def logistic_regression_L1(X_train, X_test, y_train, y_test):
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Accuracy
    accuracy = accuracy_fun(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    # Precision
    precision = precision_fun(y_test, y_pred)
    print(f'Precision: {precision}')
    # Recall
    recall = Recall_fun(y_test, y_pred)
    print(f'Recall: {recall}')
    # F1-Score
    f1_score = F1_Score_fun(y_test, y_pred)
    print(f'F1-Score: {f1_score}')
    # ROC - AUC
    auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", auc)
    return y_pred


def logistic_regression_L2(X_train, X_test, y_train, y_test):
    model = LogisticRegression(penalty='l2')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Accuracy
    accuracy = accuracy_fun(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    # Precision
    precision = precision_fun(y_test, y_pred)
    print(f'Precision: {precision}')
    # Recall
    recall = Recall_fun(y_test, y_pred)
    print(f'Recall: {recall}')
    # F1-Score
    f1_score = F1_Score_fun(y_test, y_pred)
    print(f'F1-Score: {f1_score}')
    # ROC - AUC
    auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", auc)
    return y_pred

# Part 3: Support Vector Machines (SVM)

def svm(X_train, X_test, y_train, y_test):
    # Use all features for training and testing
    X_train_all = X_train.values
    X_test_all = X_test.values

    # Train SVM with all features
    svm_model = SVC(kernel='poly', degree=3)
    svm_model.fit(X_train_all, y_train)
    y_pred = svm_model.predict(X_test_all)

    # Accuracy
    accuracy = accuracy_fun(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    # Precision
    precision = precision_fun(y_test, y_pred)
    print(f'Precision: {precision}')
    # Recall
    recall = Recall_fun(y_test, y_pred)
    print(f'Recall: {recall}')
    # F1-Score
    f1_score = F1_Score_fun(y_test, y_pred)
    print(f'F1-Score: {f1_score}')
    # ROC - AUC
    auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", auc)

    # Plot decision boundary using only the first two features
    svm_2d = SVC(kernel='poly', degree=3)
    X_train_2d = X_train.iloc[:, [0, 1]].values  # Use only the first two features
    X_test_2d = X_test.iloc[:, [0, 1]].values  # Use only the first two features
    svm_2d.fit(X_train_2d, y_train)

    # Create mesh grid for decision boundary
    plt.figure(figsize=(8, 6))
    X_set, y_set = X_train_2d, y_train.values
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, svm_2d.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape),
                 alpha=0.75, cmap=plt.get_cmap('coolwarm'))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=plt.get_cmap('coolwarm')(i / float(len(np.unique(y_set)))), label=j)
    plt.title('SVM Decision Boundary (First Two Features)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    return y_pred


# Part 4: Ensemble Methods
# 1. Boosting: Train a model using AdaBoost.
# 2. Bagging: Train a model using Bagging or Random Forest.
# 3. Compare the performance of Boosting and Bagging methods.
# 4. Discuss:
# o Which ensemble method performed better and why?
# o How do ensemble methods compare to individual models (KNN, Logistic
# Regression, SVM)? All using apis


def boosting(X_train, X_test, y_train, y_test):
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Accuracy
    accuracy = accuracy_fun(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    # Precision
    precision = precision_fun(y_test, y_pred)
    print(f'Precision: {precision}')
    # Recall
    recall = Recall_fun(y_test, y_pred)
    print(f'Recall: {recall}')
    # F1-Score
    f1_score = F1_Score_fun(y_test, y_pred)
    print(f'F1-Score: {f1_score}')
    # ROC - AUC
    auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", auc)
    return y_pred


def bagging(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Accuracy
    accuracy = accuracy_fun(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    # Precision
    precision = precision_fun(y_test, y_pred)
    print(f'Precision: {precision}')
    # Recall
    recall = Recall_fun(y_test, y_pred)
    print(f'Recall: {recall}')
    # F1-Score
    f1_score = F1_Score_fun(y_test, y_pred)
    print(f'F1-Score: {f1_score}')
    # ROC - AUC
    auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", auc)
    return y_pred

def ensemble_methods(X_train, X_test, y_train, y_test):
    print('Boosting')
    boosting(X_train, X_test, y_train, y_test)
    print('Bagging')
    bagging(X_train, X_test, y_train, y_test)



if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    encode_output_label(df, 'diagnosis')
    df = minMax_normalization(df)
    # drop the id feature as it has no effects to the output label
    df = df.drop(columns=['id'])
    # split the dataset
    X_train, X_test, y_train, y_test = split_dataSet(df, 'diagnosis')
    # k = cross_validation_k(X_train, y_train, 1, 50)
    # print('best k is: ', k)
    # knn(X_train, X_test, y_train, y_test, k)
    # # plot ROC curve
    # y_pred = knn(X_train, X_test, y_train, y_test, k)
    # roc_plot(y_test, y_pred)
    # Logistic Regression
    # print('Logistic Regression')
    # y_pred = logistic_regression_L2(X_train, X_test, y_train, y_test)
    # roc_plot(y_test, y_pred)
    # y_pred = logistic_regression(X_train, X_test, y_train, y_test)
    # roc_plot(y_test, y_pred)

    # # Logistic Regression with L1 regularization
    # print('Logistic Regression with L1 regularization')
    # logistic_regression_L1(X_train, X_test, y_train, y_test)

    # # Logistic Regression with L2 regularization
    # print('Logistic Regression with L2 regularization')
    # logistic_regression_L2(X_train, X_test, y_train, y_test)

    # # SVM
    # print('SVM')
    # svm(X_train, X_test, y_train, y_test)

    # # Ensemble Methods
    # print('Ensemble Methods')
    # ensemble_methods(X_train, X_test, y_train, y_test)
