import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
data = pd.read_csv('titanic.csv')

# Select relevant features and target variable
features = ['Pclass', 'Age', 'Sex', 'Fare']
target = 'Survived'

# Handle missing values
data = data[features + [target]].dropna()

# Convert categorical variables to numeric
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
