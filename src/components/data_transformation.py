import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        # We define exactly where to save the .pkl files for the web app
        self.scaler_obj_file_path = os.path.join('artifacts', "scaler.pkl")
        self.encoder_obj_file_path = os.path.join('artifacts', "encoder.pkl")

    def initiate_data_transformation(self, train_path, test_path):
        print("Starting Data Transformation...")
        try:
            # 1. Load the data created by the Data Ingestion script
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 2. Separate Features (X) and Target (y)
            X_train = train_df.drop('churn', axis=1)
            y_train = train_df['churn']
            X_test = test_df.drop('churn', axis=1)
            y_test = test_df['churn']

            # 3. Manual Mapping for Gender
            gender_map = {'Male': 1, 'Female': 0}
            X_train['gender'] = X_train['gender'].map(gender_map)
            X_test['gender'] = X_test['gender'].map(gender_map)

            # 4. Categorical Encoding (Country)
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            ohe_cols = encoder.fit(X_train[['country']]).get_feature_names_out(['country'])
            
            X_train_encoded = pd.DataFrame(encoder.transform(X_train[['country']]), columns=ohe_cols, index=X_train.index)
            X_train = pd.concat([X_train.drop('country', axis=1), X_train_encoded], axis=1)

            X_test_encoded = pd.DataFrame(encoder.transform(X_test[['country']]), columns=ohe_cols, index=X_test.index)
            X_test = pd.concat([X_test.drop('country', axis=1), X_test_encoded], axis=1)

            # 5. Feature Scaling
            cols_to_scale = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
            scaler = StandardScaler()
            
            X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
            X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

            # 6. Target Balancing (SMOTE) ONLY on Training Data
            print("Applying SMOTE Target Balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            # 7. Save the Translator Objects using our utils.py toolbox!
            save_object(self.scaler_obj_file_path, scaler)
            save_object(self.encoder_obj_file_path, encoder)

            print("✅ Data Transformation Complete!")

            # Return the finalized, mathematical matrices
            return (X_train_balanced, y_train_balanced, X_test, y_test)

        except Exception as e:
            print(f"Error in Data Transformation: {e}")
            raise e