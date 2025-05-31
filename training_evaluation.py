from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_classifier.fit(X_train, y_train)

# Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Validation set evaluation
classifiers = [svm_classifier, rf_classifier, gb_classifier, dt_classifier]
classifier_names = ['SVM', 'Random Forest', 'Gradient Boosting', 'Decision Tree']

for clf, name in zip(classifiers, classifier_names):
    y_val_pred = clf.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    precision_val = precision_score(y_val, y_val_pred, average='weighted')
    recall_val = recall_score(y_val, y_val_pred, average='weighted')
    conf_matrix_val = confusion_matrix(y_val, y_val_pred)

    print(f"{name} Validation Metrics:")
    print(f"Accuracy: {accuracy_val * 100:.2f}%")
    print(f"Precision: {precision_val * 100:.2f}%")
    print(f"Recall: {recall_val * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix_val)
    print()

# Test set evaluation
for clf, name in zip(classifiers, classifier_names):
    y_test_pred = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='weighted')
    recall_test = recall_score(y_test, y_test_pred, average='weighted')
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)

    print(f"{name} Test Metrics:")
    print(f"Accuracy: {accuracy_test * 100:.2f}%")
    print(f"Precision: {precision_test * 100:.2f}%")
    print(f"Recall: {recall_test * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix_test)
    print()

# Plot ROC curves for test set
plt.figure(figsize=(8, 6))
for clf, name in zip(classifiers, classifier_names):
    y_score_test = clf.decision_function(X_test) if hasattr(clf, "decision_function") else clf.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, label=f'{name} (AUC = {roc_auc_test:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Test Set')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC()
rf_model = RandomForestClassifier()
dt_model = DecisionTreeClassifier()
gb_model = GradientBoostingClassifier()

models = {'SVM': svm_model, 'Random Forest': rf_model, 'Decision Tree': dt_model, 'Gradient Boosting': gb_model}

metrics = {'Accuracy': accuracy_score,
           'Precision': precision_score,
           'Recall': recall_score,
           'F1 Score': f1_score,
           'ROC AUC': roc_auc_score}

results_df = pd.DataFrame(columns=metrics.keys())

for name, model in models.items():
    model_scores = {}
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    for metric_name, metric_func in metrics.items():
        model_scores[metric_name] = metric_func(y_test, y_pred)
    results_df = results_df.append(pd.Series(model_scores, name=name))

print("Statistical Report for Machine Learning Models:")
print(results_df)\
import matplotlib.pyplot as plt

results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Performance Metrics Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
