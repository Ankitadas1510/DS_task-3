import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LinearRegression # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
import joblib # pyright: ignore[reportMissingImports]
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "data", "student_data.csv")
model_dir = os.path.join(BASE_DIR, "model")
model_path = os.path.join(model_dir, "model.pkl")

df = pd.read_csv(data_path)

X = df[['study_hours', 'attendance', 'previous_score']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, model_path)

print("✅ Model saved successfully at:", model_path)
