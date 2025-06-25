import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('template/Crop_recommendation.csv')

# Check for missing values
if data.isnull().sum().sum() > 0:  # Check if there are any missing values
    print("Missing values found. Filling missing values...")
    data = data.fillna(data.mean())  # Replace missing values with column mean (or other strategy)
else:
    print("No missing values found.")

# Splitting features (X) and labels (y)
x = data.iloc[:, :-1]  # Features (all columns except the last)
y = data.iloc[:, -1]   # Labels (last column)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and evaluate accuracy
predictions = model.predict(x_test)
accuracy = model.score(x_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Test with new features
new_features = [[36, 34, 28, 23.45644, 39.4563, 45.6666, 43.322]]  # Example new data
predicted_crop = model.predict(new_features)
print(f"Predicted Crop: {predicted_crop}")

# Save the trained model using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

