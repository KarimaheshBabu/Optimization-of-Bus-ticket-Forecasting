import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from flask import Flask, request, jsonify

# Business Understanding:
# The goal is to predict whether a patient will develop coronary heart disease (CHD) within ten years

# Data Loading
df = pd.read_csv(r"C:\Users\Admin\Downloads\framingham.csv")
# Data Preprocessing
df.drop(columns=['education'], inplace=True)  # Dropping unnecessary column
df.rename(columns={'male': 'Sex_male'}, inplace=True)
df.dropna(inplace=True)  # Removing missing values

# Exploratory Data Analysis
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=df, palette="BuGn_r")
plt.title("Distribution of CHD Cases")
plt.show()

# Feature Selection
features = ['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']
X = df[features].values
y = df['TenYearCHD'].values

# Data Normalization
X = preprocessing.StandardScaler().fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# Model Training
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(logreg, model_file)

# Predictions
y_pred = logreg.predict(X_test)

# Model Evaluation
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Deployment using Flask
app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)


