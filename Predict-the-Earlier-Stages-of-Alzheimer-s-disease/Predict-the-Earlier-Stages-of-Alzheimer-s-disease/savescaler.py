from sklearn.preprocessing import StandardScaler
import pickle

# Example data to fit the scaler (replace with your actual data)
X_train = [[1, 2], [2, 3], [3, 4]]

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler to a file
with open("scaler.bin", "wb") as f:
    pickle.dump(scaler, f)
