import numpy as np
import matplotlib.pyplot as plt
import datetime

# Step 1: Read data from the CSV file
data = np.genfromtxt('CLF_data.csv', delimiter=',',
                     skip_header=1, dtype=None, encoding=None)

# Extracting dates and closing prices
dates = [datetime.datetime.strptime(row[0], "%Y-%m-%d") for row in data]
closing_prices = [row[4] for row in data]

# Step 2: Define sliding window length and standard deviation multiplier
moving_avg = 50

# Step 3: Compute moving average and standard deviation for window size 50
moving_averages = []


# Moving averages for window size 50
for i in range(len(closing_prices) - moving_avg + 1):
    window_prices = closing_prices[i:i + moving_avg]
    # Compute moving average
    moving_avg_price = np.mean(window_prices)
    moving_averages.append(moving_avg_price)


# Step 4: Plotting for window size 50
plt.figure(figsize=(10, 6))

# Plotting closing prices
plt.plot(dates[moving_avg-1:], closing_prices[moving_avg-1:],
         label='Closing Prices')

# Plotting moving average
plt.plot(dates[moving_avg-1:], moving_averages, color='red',
         label=f'{moving_avg}-Day Moving Average')


# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Moving Average and Bollinger Bands (Window Size = 50)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()


# Step 5: Compute moving average and standard deviation for window size 100
moving_avg = 100
moving_averages_100 = []

# Moving averages for window size 100
for i in range(len(closing_prices) - moving_avg + 1):
    window_prices = closing_prices[i:i + moving_avg]
    # Compute moving average
    moving_avg_price = np.mean(window_prices)
    moving_averages_100.append(moving_avg_price)


# Plotting moving average
plt.plot(dates[moving_avg-1:], moving_averages_100,
         color='blue', label=f'{moving_avg}-Day Moving Average')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Moving Average and Bollinger Bands (Window Size = 100)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()


moving_avg = 20
std_dev_multiplier = 2
upper_bands = []
lower_bands = []

for i in range(len(closing_prices) - moving_avg + 1):
    window_prices = closing_prices[i:i + moving_avg]

    # Compute moving average
    moving_avg_price = np.mean(window_prices)

    # Compute standard deviation
    std_dev = np.std(window_prices)

    # Compute upper and lower bands
    upper_band = moving_avg_price + std_dev_multiplier * std_dev
    lower_band = moving_avg_price - std_dev_multiplier * std_dev

    upper_bands.append(upper_band)
    lower_bands.append(lower_band)

plt.plot(dates[moving_avg-1:], upper_bands,
         color='cyan', linestyle='--', label='Upper Band')
plt.plot(dates[moving_avg-1:], lower_bands,
         color='cyan', linestyle='--', label='Lower Band')

plt.legend()

# Show all plots at once
plt.show()
