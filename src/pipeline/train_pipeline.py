import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    print("🚀 STARTING FINTECH AI FACTORY PIPELINE 🚀\n")
    try:
        # Step 1: Pull and Split the Data
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print("-" * 50)

        # Step 2: Translate to Math and Balance with SMOTE
        transformer = DataTransformation()
        X_train, y_train, X_test, y_test = transformer.initiate_data_transformation(train_path, test_path)
        print("-" * 50)

        # Step 3: Train the XGBoost Engine
        trainer = ModelTrainer()
        saved_model_path = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
        print("-" * 50)

        print(f"\n🎉 MASSIVE SUCCESS! Entire pipeline executed flawlessly.")
        print(f"📁 New production model safely stored at: {saved_model_path}")

    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        sys.exit(1)