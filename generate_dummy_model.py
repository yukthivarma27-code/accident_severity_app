import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    print("Generating dummy model data...")
    # Create dummy data
    # Features: temperature, visibility, wind_speed, weather_condition
    data = {
        'temperature': [70.5, 32.1, 85.0, 45.2, 55.4, 60.1, 90.5, 25.0, 40.0, 30.0],
        'visibility': [10.0, 2.5, 10.0, 5.0, 8.0, 9.0, 10.0, 0.5, 1.0, 2.0],
        'wind_speed': [5.0, 15.0, 2.0, 10.0, 6.0, 8.0, 4.0, 25.0, 20.0, 12.0],
        'weather_condition': ['Clear', 'Snow', 'Clear', 'Rain', 'Cloudy', 'Clear', 'Clear', 'Storm', 'Fog', 'Snow']
    }
    df = pd.DataFrame(data)

    # Dummy target: 1=Low, 2=Moderate, 3=High, 4=Severe
    y = [1, 3, 1, 2, 2, 1, 1, 4, 3, 3]

    # Map weather conditions to numeric for the dummy scikit-learn model
    # Note: If the real model uses OneHotEncoder inside the pipeline, 
    # it won't need this manual mapping. We'll use a manual map here just to make the dummy model work easily.
    weather_mapping = {
        'Clear': 0, 'Cloudy': 1, 'Rain': 2, 'Snow': 3, 'Storm': 4, 'Fog': 5
    }
    df['weather_condition'] = df['weather_condition'].map(weather_mapping).fillna(0)

    # Create a basic pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=10, random_state=42))
    ])

    # Train model
    pipeline.fit(df, y)

    # Save to disk
    joblib.dump(pipeline, 'accident_severity_model.pkl')
    print("accident_severity_model.pkl created successfully!")

if __name__ == '__main__':
    main()
