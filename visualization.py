import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ClimateVisualizer:
    def __init__(self):
        sns.set_theme()
        self.colors = sns.color_palette("husl", 3)
        
    def plot_predictions(self, actual, predictions):
        """
        Plot actual vs predicted temperatures
        """
        plt.figure(figsize=(12, 6))
        plt.scatter(actual, predictions, alpha=0.5, color=self.colors[0])
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 
                'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Temperature (°C)')
        plt.ylabel('Predicted Temperature (°C)')
        plt.title('Actual vs Predicted Temperatures')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('predictions_scatter.png')
        plt.close()
        
    def plot_temperature_trend(self, years, actual, predictions):
        """
        Plot temperature trends over time
        """
        plt.figure(figsize=(15, 7))
        plt.plot(years, actual, label='Actual', color=self.colors[0], alpha=0.7)
        plt.plot(years, predictions, label='Predicted', color=self.colors[1], 
                linestyle='--', alpha=0.7)
        
        plt.fill_between(years, actual, predictions, 
                        color=self.colors[2], alpha=0.2, label='Prediction Error')
        
        plt.xlabel('Year')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Trends: Actual vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('temperature_trends.png')
        plt.close()
        
    def plot_training_history(self, history):
        """
        Plot training history (loss and MAE)
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss', color=self.colors[0])
        plt.plot(history.history['val_loss'], label='Validation Loss', 
                color=self.colors[1], linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE', color=self.colors[0])
        plt.plot(history.history['val_mae'], label='Validation MAE', 
                color=self.colors[1], linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Model MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
    def plot_feature_importance(self, feature_names, importance_scores):
        """
        Plot feature importance scores
        """
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        
        plt.barh(y_pos, importance_scores, align='center', color=self.colors[0])
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close() 