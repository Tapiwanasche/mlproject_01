import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from src.utils import save_object

class ModelTrainer:
    def __init__(self):
        # Define exactly where the new AI brain will be saved
        self.trained_model_file_path = os.path.join("artifacts", "xgboost_model.pkl")

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        print("Starting Model Training Component...")
        try:
            # 1. Initialize the XGBoost Engine 
            # We hardcode the winning parameters we found during Phase 3
            xgb_model = XGBClassifier(
                learning_rate=0.1,
                max_depth=7,
                n_estimators=200,
                eval_metric='logloss',
                random_state=42
            )

            # 2. Train the model on the balanced data
            print("Training XGBoost Engine... (This might take a few seconds)")
            xgb_model.fit(X_train, y_train)

            # 3. Evaluate the model using our Secret Weapon (0.40 Threshold)
            print("Evaluating Model Performance...")
            y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]
            y_pred_custom = (y_pred_probs >= 0.40).astype(int)
            
            print("\n" + "="*50)
            print("--- FINAL PIPELINE CLASSIFICATION REPORT ---")
            print(classification_report(y_test, y_pred_custom))
            print("="*50 + "\n")

            # 4. Save the completed model to the artifacts folder
            save_object(file_path=self.trained_model_file_path, obj=xgb_model)
            
            print("✅ Model Training Complete and Saved!")
            
            return self.trained_model_file_path

        except Exception as e:
            print(f"Error in Model Trainer: {e}")
            raise e