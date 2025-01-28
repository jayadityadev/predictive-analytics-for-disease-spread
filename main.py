import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Convert DATE_DIED to datetime and filter out non-death rows
    df["DATE_DIED"] = pd.to_datetime(
        df["DATE_DIED"], format="%d/%m/%Y", errors="coerce"
    )
    df = df.dropna(subset=["DATE_DIED"])

    # Group by date and count deaths
    daily_deaths = df.groupby("DATE_DIED").size().reset_index(name="deaths")

    # Rename columns to match Prophet requirements
    daily_deaths = daily_deaths.rename(columns={"DATE_DIED": "ds", "deaths": "y"})

    return daily_deaths


def train_model(df):
    # Create and train the Prophet model
    model = Prophet()
    model.fit(df)
    return model


def make_predictions(model, periods):
    # Generate future dates
    future_dates = model.make_future_dataframe(periods=periods)

    # Make predictions
    forecast = model.predict(future_dates)
    return forecast


def visualize_results(model, forecast, df):
    # Plot the forecast using matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"], df["y"], label="Actual")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
    plt.fill_between(
        forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3
    )
    plt.xlabel("Date")
    plt.ylabel("Daily Deaths")
    plt.title("COVID-19 Daily Deaths Forecast")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # File path to your CSV data
    file_path = "covid19_data.csv"

    # Load and preprocess the data
    df = load_and_preprocess_data(file_path)

    # Train the model
    model = train_model(df)

    # Make predictions for the next 30 days
    forecast = make_predictions(model, periods=30)

    # Visualize the results
    visualize_results(model, forecast, df)


if __name__ == "__main__":
    main()
