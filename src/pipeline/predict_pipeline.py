import os
import pandas as pd
import joblib

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # 1. Define the paths to your frozen artifacts
            model_path = os.path.join("artifacts", "xgboost_model.pkl")
            scaler_path = os.path.join("artifacts", "scaler.pkl")
            encoder_path = os.path.join("artifacts", "encoder.pkl")

            # 2. Load the artifacts into memory
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            encoder = joblib.load(encoder_path)

            # 3. Apply the exact same encoding from Phase 2
            # Map Gender
            features['gender'] = features['gender'].map({'Male': 1, 'Female': 0})
            
            # One-Hot Encode Country
            ohe_cols = encoder.get_feature_names_out(['country'])
            encoded_country = pd.DataFrame(encoder.transform(features[['country']]), columns=ohe_cols, index=features.index)
            features = pd.concat([features.drop('country', axis=1), encoded_country], axis=1)

            # 4. Apply the exact same scaling from Phase 2
            cols_to_scale = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
            features[cols_to_scale] = scaler.transform(features[cols_to_scale])

            # 5. Make the final prediction (0 or 1)
            prediction = model.predict(features)
            return prediction

        except Exception as e:
            print(f"Error in Prediction Pipeline: {e}")
            raise e


class CustomData:
    def __init__(self, 
                 credit_score: int,
                 country: str,
                 gender: str,
                 age: int,
                 tenure: int,
                 balance: float,
                 products_number: int,
                 credit_card: int,
                 active_member: int,
                 estimated_salary: float):
        
        # Catch the raw data from the HTML form and store it
        self.credit_score = credit_score
        self.country = country
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.products_number = products_number
        self.credit_card = credit_card
        self.active_member = active_member
        self.estimated_salary = estimated_salary

    def get_data_as_data_frame(self):
        try:
            # Package the raw data into a Dictionary
            custom_data_input_dict = {
                "credit_score": [self.credit_score],
                "country": [self.country],
                "gender": [self.gender],
                "age": [self.age],
                "tenure": [self.tenure],
                "balance": [self.balance],
                "products_number": [self.products_number],
                "credit_card": [self.credit_card],
                "active_member": [self.active_member],
                "estimated_salary": [self.estimated_salary],
            }

            # Convert the Dictionary into a Pandas DataFrame (just like Notebook 2)
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            print(f"Error converting data to DataFrame: {e}")
            raise e