import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def train_model(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df[["trailer_type", "estimated_days", "parts_days", "accident_days", "training_days"]]
    y = df["actual_days"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("type", OneHotEncoder(handle_unknown="ignore"), ["trailer_type"]),
        ],
        remainder="passthrough"  # keeps the numeric columns
    )
    
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    model.fit(X, y)
    return model

def predict_days(model, trailer_type, estimated_days, parts_days=0, accident_days=0, training_days=0):
    X_new = pd.DataFrame([{
        "trailer_type": trailer_type,
        "estimated_days": estimated_days,
        "parts_days": parts_days,
        "accident_days": accident_days,
        "training_days": training_days
    }])
    pred = model.predict(X_new)[0]
    return max(0, round(float(pred), 1))
