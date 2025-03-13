# Optimization-of-Bus-ticket-Forecasting
# Bus Ticket Revenue Forecasting App

## 1. Business and Data Understanding

### Project Overview

This project aims to optimize bus ticketing demand forecasting using machine learning and deep learning models. The application predicts future revenue using ARIMA, SARIMA, and LSTM models based on historical ticket sales data. The deployment is handled through a Streamlit web application.

### Business Problem

- Inefficient resource allocation leads to revenue loss and poor customer experience.
- The goal is to enhance prediction accuracy to optimize bus schedules and revenue.

### Success Criteria

- **Business:** At least a 15% improvement in customer satisfaction.
- **ML:** Achieve a Mean Absolute Percentage Error (MAPE) of less than 10%.
- **Economic:** Reduce operational costs by at least 15%.

### Dataset

The dataset contains 14,278 rows and 13 columns. It includes the following key features:

- `Date`: The date of operation (must be in a valid date format).
- `Revenue Generated (INR)`: The revenue earned per day.
- `Trips per Day`: Number of trips operated per day.
- `Bus Route No.`: The route number for each trip (**17 missing values**).
- `From`: The starting location of the trip.
- `To`: The destination of the trip.
- `Way`: Direction of travel.
- `Main Station`: Main hub for the route.
- `Frequency (mins)`: Time interval between buses (**16 missing values**).
- `Distance Travelled (km)`: Distance covered per trip (**19 missing values**).
- `Time (mins)`: Travel duration (**30 missing values**).

## 2. Data Preparation

- Convert `Date` column to datetime format.
- Handle missing values in numerical and categorical columns.
- Apply feature scaling using MinMaxScaler.
- Encode categorical variables.
- Create lag features for better time series forecasting.

## 3. Model Building

### Models Used

- **ARIMA:** Autoregressive Integrated Moving Average for time series forecasting.
- **SARIMA:** Seasonal ARIMA for capturing seasonality.
- **LSTM:** Deep learning-based model for sequential predictions.

### Training

- Train models on preprocessed data.
- Use appropriate hyperparameters and validation techniques.
- Evaluate models using MAE, RMSE, and MAPE.

## 4. Model Evaluation

### Metrics Used

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**

Models are compared based on the above metrics to select the best-performing one.

## 5. Model Deployment

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bus-revenue-forecasting.git
cd bus-revenue-forecasting
```

#### 2. Install Dependencies

Ensure you have Python 3.8+ installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

#### 3. Run the Application

```bash
streamlit run app.py
```

## 6. Monitoring and Maintenance

- Regularly update models with fresh data.
- Monitor model performance using real-time data.
- Retrain models periodically to maintain accuracy.

## Model Files

Before running the app, ensure the following pre-trained model files are available in the project directory:

- `sarima_model.pkl`
- `arima_model.pkl`
- `lstm_model.h5`

If you haven't trained the models yet, refer to the model training script (`train_models.py`) to generate these files.

## Usage

1. Open the Streamlit app.
2. Upload a CSV dataset.
3. Select the forecasting model (SARIMA, ARIMA, or LSTM).
4. Choose the number of days for forecasting.
5. View the predictions and visualization.

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- statsmodels
- tensorflow
- matplotlib
- seaborn
- joblib
- streamlit

To install all dependencies, use:

```bash
pip install -r requirements.txt
```

## Deployment

To deploy this Streamlit app on a cloud platform like Streamlit Sharing, AWS, or Heroku:

1. Push the repository to GitHub.
2. Set up Streamlit Community Cloud ([https://streamlit.io/cloud](https://streamlit.io/cloud)).
3. Deploy using the app entry point `app.py`.

## Author

Kari Mahesh Babu.

## License

This project is licensed under the MIT License.

