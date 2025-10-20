
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the Dataset

data = pd.read_csv(r"C:\Users\fatai\Desktop\ML AND AI Learning\StudentPerformanceFactors.csv")  #Update with your path

# Display first few rows
print("Dataset Preview:")
print(data.head())

# Check info
print("\n Dataset Info:")
print(data.info())

# Data Cleaning

# Check for missing values
print("\n Missing values in each column:\n", data.isnull().sum())

# Drop rows with missing values (if any)
data.dropna(inplace=True)

# Check basic statistics
print("\n Statistical Summary:")
print(data.describe())


# Data Visualization

# Visualize relationship between Study Hours and Exam Score
plt.figure(figsize=(7,5))
sns.scatterplot(x='Hours_Studied', y='Exam_Score', data = data, color='teal')
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.show()



# Prepare Training Data
X = data[['Hours_Studied']]   
y = data['Exam_Score']    

# Spliting into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
print("\n Model Coefficient (Slope):", model.coef_[0])
print(" Model Intercept:", model.intercept_)


# Make Predictions
y_pred = model.predict(X_test)

# Comparing actual vs predicted
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\n Actual vs Predicted Scores:")
print(comparison.head())


# Visualization of Predictions

plt.figure(figsize=(7,5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Actual vs Predicted Exam Scores')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.show()


#  Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# Predict a Custom Input
# Check: Predict score for a student who studied for specified hours
hours = np.array([[float(input("Enter number of study hours: "))]])
predicted_score = model.predict(hours)[0]
print(f"\n Predicted Exam Score for 8 Study Hours: {predicted_score:.2f}")
