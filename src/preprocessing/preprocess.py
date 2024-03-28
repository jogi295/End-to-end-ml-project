import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def preprocess(self, data_path, output_path):
        # Load the dataset
        data = pd.read_csv(data_path)
        X = data.drop('coordinate', axis=1)
        y = data['coordinate']
        
        # Split the data into training, testing, and evaluation sets
        X_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        X_train, X_eval, y_train, y_eval = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

        # Normalize the data
        scaler = MinMaxScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        X_eval_normalized = scaler.transform(X_eval)

        # Save the normalized data to CSV files
        pd.DataFrame(X_train_normalized, columns=X_train.columns).to_csv(output_path + '/X_train_normalized.csv', index=False)
        pd.DataFrame(X_test_normalized, columns=X_test.columns).to_csv(output_path + '/X_test_normalized.csv', index=False)
        pd.DataFrame(X_eval_normalized, columns=X_eval.columns).to_csv(output_path + '/X_eval_normalized.csv', index=False)

        # Save labels to CSV files
        pd.DataFrame(y_train, columns=['coordinate']).to_csv(output_path + '/y_train.csv', index=False)
        pd.DataFrame(y_test, columns=['coordinate']).to_csv(output_path + '/y_test.csv', index=False)
        pd.DataFrame(y_eval, columns=['coordinate']).to_csv(output_path + '/y_eval.csv', index=False)

        with open('./models/preprocessor.pkl', 'wb') as f:
            pickle.dump(scaler, f)



# Usage:
if __name__ == "__main__":
    data_path = './data/raw/data.csv'
    output_path = './data/processed'
    
    print("Preprocessing started::")
    preprocessor = Preprocessor()

    preprocessor.preprocess( data_path, output_path )
    print("Preprocessing completed::")
