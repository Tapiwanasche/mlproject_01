import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        # We tell the script exactly where to save the new CSV files
        self.train_data_path = os.path.join('artifacts', "train.csv")
        self.test_data_path = os.path.join('artifacts', "test.csv")
        self.raw_data_path = os.path.join('artifacts', "data.csv")

    def initiate_data_ingestion(self):
        print("Starting Data Ingestion Component...")
        try:
            # 1. Read the raw data (simulating pulling from a bank's database)
            # Adjust the path if your data is located somewhere else
            df = pd.read_csv('data/bank_churn.csv')
            
            # Drop the useless ID column immediately
            df = df.drop(['customer_id'], axis=1)

            # 2. Save a copy of the raw data to artifacts
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False, header=True)

            print("Running Train-Test Split...")
            # 3. Perform the Stratified Split
            X = df.drop(['churn'], axis=1)
            y = df['churn']
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

            # 4. Save the train and test sets directly to the artifacts folder
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)

            print("✅ Data Ingestion Complete!")
            
            # Return the paths so the next script knows where to find the data
            return (self.train_data_path, self.test_data_path)

        except Exception as e:
            print(f"Error in Data Ingestion: {e}")
            raise e