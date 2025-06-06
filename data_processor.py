import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import re

class ClimateDataProcessor:
    def __init__(self):
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_climate_data(self, file_path='data.csv'):
        """
        Load climate data from local CSV file
        Returns a pandas DataFrame with the climate data
        """
        try:
            # Read the data, skipping the header rows and using fixed-width format
            with open(file_path, 'r') as f:
                # Skip the first 7 lines (header)
                for _ in range(7):
                    next(f)
                
                # Read the data lines
                data_lines = []
                for line in f:
                    if line.strip() and not line.startswith('Year'):  # Skip empty lines and header repeats
                        data_lines.append(line)
            
            # Process the data lines
            processed_data = []
            year_pattern = re.compile(r'^\d{4}$')
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 19 and year_pattern.match(parts[0]):  # Only process lines starting with a 4-digit year
                    year = int(parts[0])
                    monthly_temps = [float(x) if x != '****' else np.nan for x in parts[1:13]]
                    yearly_avg = float(parts[13]) if parts[13] != '****' else np.nan
                    processed_data.append({
                        'Year': year,
                        'Jan': monthly_temps[0],
                        'Feb': monthly_temps[1],
                        'Mar': monthly_temps[2],
                        'Apr': monthly_temps[3],
                        'May': monthly_temps[4],
                        'Jun': monthly_temps[5],
                        'Jul': monthly_temps[6],
                        'Aug': monthly_temps[7],
                        'Sep': monthly_temps[8],
                        'Oct': monthly_temps[9],
                        'Nov': monthly_temps[10],
                        'Dec': monthly_temps[11],
                        'Yearly_Avg': yearly_avg
                    })
            
            # Convert to DataFrame
            data = pd.DataFrame(processed_data)
            
            # Convert temperature columns to actual temperature changes (divide by 100)
            temp_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Yearly_Avg']
            
            for col in temp_columns:
                data[col] = data[col] / 100.0
            
            # Drop rows with missing values
            data = data.dropna()
            
            print(f"Loaded {len(data)} rows of climate data.")
            self.data = data
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def prepare_data(self, sequence_length=10, target_column='Yearly_Avg'):
        """
        Prepare the data for training by creating sequences
        sequence_length: number of years to use for prediction
        target_column: column to use as target variable
        """
        if self.data is None:
            raise ValueError("No data available. Please load data first.")
            
        # Create sequences
        X, y = [], []
        for i in range(len(self.data) - sequence_length):
            X.append(self.data[target_column].iloc[i:(i + sequence_length)].values)
            y.append(self.data[target_column].iloc[i + sequence_length])
            
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Scale the data with separate scalers
        X = self.x_scaler.fit_transform(X)
        y = self.y_scaler.fit_transform(y)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def inverse_transform_y(self, data):
        """
        Convert scaled target data back to original scale
        """
        return self.y_scaler.inverse_transform(data)

    def inverse_transform(self, data):
        """
        Convert scaled data back to original scale
        """
        return self.scaler.inverse_transform(data)
    
    def get_feature_names(self):
        """
        Return the names of the features used in the model
        """
        return ['Temperature']

if __name__ == "__main__":
    processor = ClimateDataProcessor()
    data = processor.load_climate_data()
    if data is not None:
        print("First 5 rows of processed data:")
        print(data.head())
        X_train, X_test, y_train, y_test = processor.prepare_data()

        # Train a regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")

        # Visualize predictions vs actual
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Scaled Yearly Avg Temp')
        plt.legend()
        plt.title("Actual vs Predicted Yearly Avg Temperature")
        plt.show()
        plt.savefig("prediction_plot.png")
    else:
        print("Data could not be loaded.")

