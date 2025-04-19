#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

#LOADING DATASET
iris = load_iris()
iris

#CREATING DATASET TO DATAFRAME AND DEFINING TARGET NAMES
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map(dict(zip(range(3), iris.target_names)))
df.head()

#DEFINING TRAINING AND TESTING SAMPLES
X = df[iris.feature_names]
y = df['target']

#DISPLAYING TRAINING AND TESTING SAMPLES
X
y

#DEFINING test_size BY train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

#CALLING DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)

#DISPLAYING ACCURACY AND CLASSIFICATION REPORT
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

#DISPLAYING CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#DISPLAYING THE FINAL DECISIONTREECLASSIFIER OUTPUT
plt.figure(figsize=(16,10))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree Visualization (Iris Dataset)", fontsize=16)
plt.show()
