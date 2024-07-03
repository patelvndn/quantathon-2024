import numpy as np
import matplotlib.pyplot as plt
import datetime

# Step 1: Read data from the CSV file
data = np.genfromtxt('CLF_data.csv', delimiter=',',
                     skip_header=1, dtype=None, encoding=None)

# Extracting dates and closing prices
dates = [datetime.datetime.strptime(row[0], "%Y-%m-%d") for row in data]
global_dates = [datetime.datetime.strptime(row[0], "%Y-%m-%d") for row in data]

closing_prices = [row[4] for row in data]
alpha = {}


def moving_avg(window, clr='blue'):
    moving_avg = window
    moving_averages = []

    # Moving averages for window size 50
    for i in range(len(closing_prices) - moving_avg + 1):
        window_prices = closing_prices[i:i + moving_avg]
        # Compute moving average
        moving_avg_price = np.mean(window_prices)
        moving_averages.append(moving_avg_price)

    # Plotting moving average
    plt.plot(dates[moving_avg-1:], moving_averages, color=clr,
             label=f'{moving_avg}-Day Moving Average')

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Moving Average and Bollinger Bands (Window Size = {window})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()


def bollinger_bands(window, std_dev_multiplier):
    moving_avg = window
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


# Plotting moving averages for different window sizes
plt.figure(figsize=(10, 6))
plt.plot(dates, closing_prices, label='Closing Prices')
moving_avg(50, 'red')
moving_avg(100, 'blue')
plt.show()

# Plotting Bollinger Bands with moving average window size 20
plt.plot(dates, closing_prices,
         label='Closing Prices')
moving_avg(20, 'red')
bollinger_bands(20, 2)
plt.show()

p = [row[4] for row in data]
# Moving averages
ma50 = [np.mean(p[i-49:i]) for i in range(50, len(p))][50:]
ma100 = [np.mean(p[i-99:i]) for i in range(100, len(p))]

# dates and prices for the last 100 days
d1 = dates[100:]
p1 = p[100:]


# LONG
# buys = {}
# buyslist = []
# sells = {}
# sellslist = []
# dateslist = []
# current = True
# condition1 = False

# for i in range(20, len(p1)):
#     # bollinger band definition
#     bt = p1[i]+2*np.std(p1[i-20:i])
#     bl = p1[i]-2*np.std(p1[i-20:i])

#     # IF entering the market / exiting the market
#     if current:
#         if ma50[i-2] > ma50[i-1] and ma50[i-1] < ma50[i]:
#             condition1 = True
#         if condition1:
#             if ma100[i-1] < ma100[i] and p1[i] > ma100[i]:
#                 buys[d1[i]] = round(p1[i], 3)
#                 dateslist.append(d1[i])
#                 buyslist.append(round(p1[i], 3))
#                 current = False
#     else:
#         if p1[i] >= bt or p1[i] <= buyslist[-1]*0.98 or p1[i] < ma50[i] or ma50[i] < ma50[i-1]:
#             current = True
#             condition1 = False
#             sells[d1[i]] = d1[i]
#             sellslist.append(round(p1[i], 3))


# Calculating returns
# total = 10000
# for (x, y, d) in zip(buyslist, sellslist, dateslist):
#     total *= 1+round((y-x)/x, 3)
#     alpha[d] = 1+round((y-x)/x, 3)
# print(round(total, 3))


# SHORT
shorts = {}
shortslist = []
covers = {}
coverlist = []
dateslist_s = []
sellsdatelist_s = []
current_s = True
condition1_s = False
for i in range(20, len(p1)):
    bt = p1[i]+2*np.std(p1[i-20:i])
    bl = p1[i]-2*np.std(p1[i-20:i])
    if current_s:
        if ma50[i-2] < ma50[i-1] and ma50[i-1] > ma50[i]:
            condition1_s = True
        if condition1_s:
            if ma100[i-1] > ma100[i] and p1[i] < ma100[i]:
                shorts[d1[i]] = round(p1[i], 3)
                dateslist_s.append(d1[i])
                shortslist.append(round(p1[i], 3))
                current_s = False
    else:
        if p1[i] <= bl or p1[i] >= shortslist[-1]*1.02 or p1[i] > ma50[i] or ma50[i] > ma50[i-1]:
            current_s = True
            condition1_s = False
            covers[d1[i]] = d1[i]
            coverlist.append(round(p1[i], 3))
            sellsdatelist_s.append(d1[i])

total_s = 10000
for (x, y, d, sd) in zip(shortslist, coverlist, dateslist_s, sellsdatelist_s):
    #     print(d,round((x-y)/y,3)*100, sd)
    total_s *= 1+round((x-y)/y, 3)
    alpha[d] = 1+round((x-y)/y, 3)
print(round(total_s, 3))
# print(alpha)

total_f = 10000

dates = [d1[0]]
cumulative_returns = [10000]

alpha_sorted = sorted(alpha.items(), key=lambda x: x[0])

for date, return_value in alpha_sorted:
    total_f *= return_value
    dates.append(date)
    cumulative_returns.append(total_f)

print(round(total_f, 2))

figs, axs = plt.subplots(1, 2, figsize=(15, 6))


# Plotting our
axs[0].plot(dates, cumulative_returns, marker='o',
            linestyle='-', label='Profits')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Portfolio Total ($)')
axs[0].set_title('Portfolio Profits')
axs[0].tick_params(axis='x', rotation=45)
axs[0].grid(True)
axs[0].legend()

# Plotting on the second subplot
axs[1].plot(global_dates, closing_prices, marker='o',
            linestyle='-', label='Closing Prices')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Price')
axs[1].set_title('Closing Prices')
axs[1].tick_params(axis='x', rotation=45)
axs[1].grid(True)
axs[1].legend()


plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(dateslist_s, shortslist, marker='o', label='Entry', color='green')
plt.scatter(sellsdatelist_s, shortslist, marker='x',
            label='Exit', color='red')
# plt.plot(dateslist, sellslist, label='Buys')
plt.plot(global_dates, closing_prices, label='Closing Prices',
         color='blue', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Shorts and Buys')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Data Calculation

# Sharpe Ratio
daily_returns = np.diff(cumulative_returns) / cumulative_returns[:-1]


risk_free_rate = 0.0527 / 252
avg_daily_return = np.mean(daily_returns)

std_dev_daily_return = np.std(daily_returns)


sharpe_ratio = (avg_daily_return - risk_free_rate) / std_dev_daily_return
# annualizing
sharpe_ratio *= np.sqrt(252)

print("Sharpe Ratio", round(sharpe_ratio, 2))

# Total Return
print('Total Return of Strat', round(
    ((cumulative_returns[-1]/cumulative_returns[0])-1)*100, 2), '%')
print('Total Return of CLF', round(p1[-1]/p1[0]*100, 2), '%')


# Avg Gain to Loss
gains = np.mean([x-1 for x in alpha.values() if x > 1])
losses = np.mean([x-1 for x in alpha.values() if x < 1])
print('Avg gain to loss ratio', round(gains/-losses, 3))

# Max Drawdown
max_drawdown = 0
peak_value = cumulative_returns[0]
for value in cumulative_returns:
    if value > peak_value:
        peak_value = value
    drawdown = (peak_value - value) / peak_value
    if drawdown > max_drawdown:
        max_drawdown = drawdown
max_drawdown_percentage = max_drawdown * 100

print("Max Drawdown of Strat", round(max_drawdown_percentage, 2), "%")

# Max Drawdown of CLF
max_drawdown_clf = 0
peak_value_clf = p1[0]
for value in p1:
    if value > peak_value_clf:
        peak_value_clf = value
    drawdown = (peak_value_clf - value) / peak_value_clf
    if drawdown > max_drawdown_clf:
        max_drawdown_clf = drawdown
max_drawdown_percentage_clf = max_drawdown_clf * 100

print("Max Drawdown of CLF", round(max_drawdown_percentage_clf, 2), "%")
