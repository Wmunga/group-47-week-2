import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from data_processor import ClimateDataProcessor
from visualization import ClimateVisualizer
import matplotlib.pyplot as plt

class ClimatePredictor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.data_processor = ClimateDataProcessor()
        self.model = None
        self.visualizer = ClimateVisualizer()
        
    def build_model(self):
        """
        Build and compile the neural network model
        """
        model = Sequential([
            LSTM(128, input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train_model(self, epochs=100, batch_size=16):
        """
        Train the model on the prepared data
        """
        if self.model is None:
            self.build_model()
            
        # Load and prepare data
        self.data_processor.load_climate_data()
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data(
            sequence_length=self.sequence_length,
            target_column='Yearly_Avg'
        )
        
        # Reshape data for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        X = X.reshape((X.shape[0], X.shape[1], 1))
        predictions = self.model.predict(X)
        return self.data_processor.inverse_transform(predictions)
    
    def evaluate_model(self):
        """
        Evaluate the model on test data
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        X_test = self.data_processor.X_test.reshape(
            (self.data_processor.X_test.shape[0], 
             self.data_processor.X_test.shape[1], 1)
        )
        
        test_loss, test_mae = self.model.evaluate(X_test, self.data_processor.y_test)
        print(f"\nTest MAE: {test_mae:.4f}")
        
        # Make predictions
        predictions = self.predict(self.data_processor.X_test)
        actual = self.data_processor.inverse_transform(self.data_processor.y_test)
        
        # Get years for x-axis
        years = self.data_processor.data['Year'].values[-len(actual):]
        
        # Visualize results
        self.visualizer.plot_predictions(actual, predictions)
        self.visualizer.plot_temperature_trend(years, actual, predictions)
        
        return test_loss, test_mae

def main():
    # Initialize and train the model
    predictor = ClimatePredictor(sequence_length=10)  # Using 10 years of data to predict next year
    history = predictor.train_model(epochs=100, batch_size=16)
    
    # Evaluate the model
    predictor.evaluate_model()
    
    # Plot training history
    predictor.visualizer.plot_training_history(history)

if __name__ == "__main__":
    main() 