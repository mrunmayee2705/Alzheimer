from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle

# Example data to train the models (replace with your actual training data)
X_train = [[1, 2], [2, 3], [3, 4]]
y_train = [0, 1, 0]

# Train Linear SVM
model1 = SVC(probability=True)
model1.fit(X_train, y_train)

# Save Linear SVM
with open("LinearSVM.bin", "wb") as f:
    pickle.dump(model1, f)

# Train Logistic Regression
model2 = LogisticRegression()
model2.fit(X_train, y_train)

# Save Logistic Regression
with open("LogisticRegression.bin", "wb") as f:
    pickle.dump(model2, f)
