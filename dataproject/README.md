# Data analysis project

**Stock Return and borrowing cost** 

The **results** of the project can be seen from running [Dataproject -- SMP -- Final Version]

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires:

     #!pip install yfinance
     #!pip install pandas-datareader
     #from statsmodels.tsa.holtwinters import ExponentialSmoothing
     #from datetime import datetime
     #import statsmodels.api as sm

**Summary**

The project involves importing and analyzing market data for the Dow Jones Industrial Average (DOW), S&P 500 (SP500), and Nasdaq (IXIC) stock indexes, as well as the US 10-year Treasury yield (TNX). The project starts with importing the necessary libraries, downloading the market data, and plotting the index prices in absolute values. The data is then cleaned, indexed, and plotted again to compare the returns of the three indexes since 1993. The project concludes by stating that from 1996 to 2023, each of the three stock indexes had positive returns with large fluctuations around the Dot-com bubble (2000-2002), Financial Crisis (2007), and the Corona Pandemic, with the Nasdaq index having the highest return over the entire period and outperforming the other indices in the sub-periods after 2006.

The project further examined the relationship between stock market returns for the Dow Jones, S&P 500, and Nasdaq indices and changes in the interest rates. We found that all three indices had a negative relationship with interest rates, indicating that an increase in the interest rate leads to a decrease in returns for these indices. Which were in line with our expectations from economic and financial theory. The results from the regression analysis were statistically significant
