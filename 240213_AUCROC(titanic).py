# Calculate AUC_ROC score using titanic dataset

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load titanic dataset
titanic = sns.load_dataset('titanic')

# Preprocess the data
titanic['sex'] = titanic['sex'].map({'male': 1, 'female': 0})
titanic['age'].fillna(titanic['age'].mean(), inplace=True)
titanic['embarked'] = titanic['embarked'].map({'C': 2, 'Q': 1, 'S': 0})

# Make dataset
X = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = titanic['survived']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression model
model = LogisticRegression(solver='liblinear')

# Fit the model
model.fit(X_train, y_train)

# Predict the test data
y_score = model.predict_proba(X_test)[:, 1]

# Calculate AUC_ROC score
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)

# Visualize ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # random guessing
plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Titanic Survival Prediction)')
plt.legend(loc='best')
plt.show()
