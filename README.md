# quantathon-2024

## Overview:

This is Theta Decays submission for the OSU Quantathonn 2024
Members: Connor R, Moksha D, Vandan P

This Python script analyzes financial data from a CSV file, computes various financial indicators, and visualizes the results using Matplotlib. It performs moving average analysis, Bollinger Bands analysis, and evaluates trading strategies based on specified conditions.

## Dependencies:

- `numpy`: For numerical operations and data manipulation.
- `matplotlib`: For creating plots and visualizations.
- `datetime`: For handling date and time data.

## Usage:

1. Ensure that the CSV file containing the financial data (`CLF_data.csv/DAL_data.csv`) is in the same directory as the script.
2. Run the Python script (`main.py`).

## Description:

- The script first reads the financial data from the CSV file and extracts relevant information such as dates and closing prices.
- It defines functions for calculating moving averages and Bollinger Bands, and plots the results.
- It then implements trading strategies based on specified conditions, calculates profits and losses, and visualizes the trading signals and closing prices.
- Finally, it computes various financial metrics such as Sharpe Ratio, Total Return, Average Gain to Loss ratio, and Maximum Drawdown for evaluating the trading strategies.

## Results:

- The script generates multiple plots showcasing moving averages, Bollinger Bands, trading signals, and closing prices.
- It calculates and prints key financial metrics to evaluate the performance of the trading strategies.

## Note:

- Make sure to customize the script parameters and conditions based on specific financial data and trading strategies.
- Additional analysis or modifications can be made to enhance the functionality and accuracy of the script.
