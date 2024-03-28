#pip install mysql-connector-python
# use above code if not mysql connector in not installed
import mysql.connector
import pandas as pd
import os

class DataIngestion:
    def __init__(self, folder_path, host, user, password, database, table):
        self.folder_path = folder_path
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table

    def ingest_data(self):
        try:
            # Connect to the MySQL database
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            cursor = connection.cursor()

            # Execute a query to fetch data from the specified table
            query = f"SELECT * FROM {self.table}"
            cursor.execute(query)

            # Fetch all the rows
            data = cursor.fetchall()

            # Get column names from the cursor description
            column_names = [desc[0] for desc in cursor.description]

            # Create a DataFrame from the fetched data
            df = pd.DataFrame(data, columns=column_names)

            # Store the DataFrame as CSV in the specified folder
            os.makedirs(self.folder_path, exist_ok=True)
            file_path = os.path.join(self.folder_path, 'data.csv')
            df.to_csv(file_path, index=False)

            print("Data ingestion completed successfully.")

        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL: {e}")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    folder_path = './data/raw_data'
    host = 'localhost'
    user = 'jogesh'
    password = 'Jogesh_295'
    database = 'coordinates_data'
    table = 'coordinates_table'

    data_ingestion_instance = DataIngestion(folder_path, host, user, password, database, table)
    data_ingestion_instance.ingest_data()


