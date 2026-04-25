import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt


df = pd.read_csv('./csv/internship_candidates_final_numeric.csv')

X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y = df['Accepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

test = pd.DataFrame({
    "Experience": [3],
    "Grade": [8.5],
    "EnglishLevel": [2],
    "Age": [24],
    "EntryTestScore": [700]
})

prediction = model.predict(test)
print(prediction)


y_pred = model.predict(X_test)

plt.scatter(X_test['EnglishLevel'], X_test['EntryTestScore'],
            c=y_pred, cmap='coolwarm', edgecolor='k', s=100)

plt.title('Logistic Regression Predictions')
plt.xlabel('English Level')
plt.ylabel('Entry Test Score')
plt.colorbar(label='Predicted Class')

plt.show()