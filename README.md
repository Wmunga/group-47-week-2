# Climate Change Prediction Model

This project implements a machine learning solution to address UN Sustainable Development Goal 13: Climate Action. It uses a neural network to predict temperature trends and extreme weather events based on historical climate data.

## Project Overview

The model uses supervised learning with a neural network architecture to:
- Predict temperature trends based on historical climate data
- Identify patterns in climate change
- Provide insights for climate action and adaptation strategies

## Features

- Data preprocessing and normalization
- Neural network model for temperature prediction
- Model evaluation and validation
- Visualization of predictions and trends
- Climate change impact assessment

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python climate_predictor.py
```

2. The script will:
   - Download and preprocess climate data
   - Train the neural network model
   - Generate predictions
   - Create visualizations

## Project Structure

- `climate_predictor.py`: Main model implementation
- `data_processor.py`: Data loading and preprocessing
- `utils.py`: Helper functions
- `visualization.py`: Plotting and visualization tools
- `requirements.txt`: Project dependencies

## Model Architecture

The neural network model consists of:
- Input layer: Historical climate data features
- Hidden layers: Multiple dense layers with ReLU activation
- Output layer: Temperature predictions
- Loss function: Mean Squared Error
- Optimizer: Adam

## Data Sources

The model uses historical climate data from reliable sources, including:
- Temperature records
- Atmospheric CO2 levels
- Sea level measurements
- Other relevant climate indicators

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UN Sustainable Development Goals
- Climate science community
- Open-source machine learning community 