#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:14:11 2025

@author: feliciaputrilawana
"""

import yfinance as yf
import numpy as np
import math
from scipy.stats import norm
from math import exp
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons, Slider
from datetime import datetime

"""
run (%matplotlib qt)  in the console to allow for an interactive plot, remove the parentheses
"""
#https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
#Documentation:
#what your code does and how to use it 

# where to change inputs to adjust the whole code

#this should be manual 

#reflective account
# make a reflective account, think about what you have learnt, what problems did i encounter a
# and what problems , what resources did i turn to and to expand my proficiency

# how would you change ur approach to better plan ahead

"""
#ADJUST TICKER AS PREFERRED
"""

tckr = 'AMZN'

#%%

#GETTING VOLATILITY & S0

#%%
#(I)
def get_volatility_S0(tckr, trading_days=252):

    """
        Downloads recent data for a ticker (variable tckr), calculates log returns,
        daily volatility, annualized volatility, and returns the last closing price (S0).
        
        Parameters:
           ticker (str): use tckr and change the variable as needed (line22)
           trading_days (int): Number of trading days to consider (default 252).
            
       Returns:
           price_df (pd.DataFrame): DataFrame with price data and log returns.
           S0 (float): Last closing price.
           vol_daily (float): Daily volatility.
           vol_annual (float): Annualized volatility.

    """
    
    # Download 2 years of daily data from yfinance
    data = yf.download(tckr, period="2y", interval="1d", auto_adjust=False)

    # Take last trading days
    price_df = data.tail(trading_days).copy()

    # Flatten multi-index columns (if any)
    price_df.columns = [
        '_'.join(col).strip() if isinstance(col, tuple) else col
        for col in price_df.columns
    ]

    # Identify Close column safely
    close_candidates = [c for c in price_df.columns if "Close" in c and "Adj" not in c]
    if not close_candidates:
        raise ValueError("No valid 'Close' column found.")
    close_col = close_candidates[0]

    # Last closing price
    S0 = price_df[close_col].iloc[-1]
    
    #had a warning that I was modifying a view so need to place loc on the code
    # Log returns
    price_df.loc[:, 'Log return'] = np.log(
        price_df[close_col] / price_df[close_col].shift(1)
    )

    # Drop NA
    price_df = price_df.dropna(subset=["Log return"])

    # Daily and annual volatility
    vol_daily = price_df["Log return"].std()
    vol_annual = vol_daily * np.sqrt(252)

    return price_df, S0, vol_daily, vol_annual
#%%
price_df, S0, vol_daily, vol_annual = get_volatility_S0(tckr, trading_days=252)

print(S0)
print(vol_annual)

#%%
'''
#BLACK SCHOLES MERTON FUNCTION (analytical method)
'''
#(I)
def BSM_price(S0, K, T, r, sigma, option_type='call'):
    
    """ 
    Calculates Option Price based on the Black scholes 
    merton model and returns Option Price 
    
    Parameters:
        S0 : Current stock price
        K : Strike Price
        r : risk-free rate
        T : length of the contract
        sigma : calculated annual volatility
        option_type : 'call' or 'put'
        
    The variables should be set on the previous block with -
    variable 'setting' being the option type
    """
    S0 = np.array(S0)  # ensures S0 is an array, even if scalar

    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    #turns option type value to lowercase
    option_type = option_type.lower()
    
    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    elif option_type == 'put':
        
        return K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    #executed if input is not expected must be call or put
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
#%%
'''
#BINOMIAL PRICING FUNCTION
'''
#(I)
#references : https://www.codearmo.com/python-tutorial/options-trading-binomial-pricing-model
trees ={}

def binomialpricing(S0, K, T, r, sigma, node, option_type = 'call'):
    """ 
    Calculates the Option price through the Binomial Pricing model 
    
    Parameters:
        S0 : Current stock price
        K : Strike Price
        r : risk-free rate
        T : length of the contract
        sigma : calculated annual volatility
        option_type : 'call' or 'put'
        node : number of time steps the function should operate through
    
    Returns the price of the option 
    on the price tree the position of the option price is [0,0]
    
    """
    # Step size in time
    dt = T / node

    # Up and down factors
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u  # For simplicity, we use the reciprocal of u as the down factor
        
    # Risk-neutral probability
    p = (math.exp(r * dt) - d) / (u - d)
    
    #remember (rows, cols)
    
    payofftree = np.zeros((node+1, node+1))
    
    payofftree[0,0] = S0
    
    #turns option_type value to lowercase
    option_type = option_type.lower()
    
    for time in range(1,node+1):
        payofftree[time,0] = payofftree[time-1,0] * d
        for move in range(1, time+1):
            payofftree[time,move] = payofftree[time-1,move-1] * u
   
    pricetree = np.zeros((node+1,node+1))
    
    if option_type == 'call':
        for move in range (0,node+1):
            pricetree[node,move] = max(0,payofftree[node,move]-K)
    
    elif option_type == 'put':
        for move in range (0,node+1):
            pricetree[node,move] = max(0,K-payofftree[node,move])
    
    #executed if input is not expected must be call or put
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    for time in range (node-1,-1,-1):
        for move in range(0, time+1):
            pricetree[time,move] = (p*pricetree[time+1,move+1] + 
                                    (1-p)*pricetree[time+1, move]) *exp(-r*dt)
    trees["payoff"] = payofftree
    trees["price"] = pricetree
    return pricetree[0,0]
#%%

#------------------------------------------------------------------------------
"""
#SET YOUR OPTION PARAMETERS HERE
"""
sigma = vol_annual
node = 500 # adjust as needed this is simplified for the development process
K = S0+50 # adjust as needed this is simplified for the development process
r = 0.05
T= 1
setting = 'call' # set the optiontype (between call or put)

#checking the values are consistent
print(S0,'\n',K,'\n',T,'\n',r,'\n',sigma,'\n',node)
#------------------------------------------------------------------------------

#%%
'''
checking the functions work 
comparing binomial and BSM price with different number of nodes
'''
#setting the option price as variables
Binomial_price = binomialpricing(S0, K, T, r, sigma, node, option_type = setting) 
BSMprice = BSM_price(S0, K, T, r, sigma, option_type = setting) 

payofftree = trees['payoff']
pricetree = trees['price']

print(Binomial_price) 

print(BSMprice)

#test set 1
#413.71837760012033 (node=1000) , Binomial price
#413.7590581317186 (node=10000), Binomial price
#413.7645735378901 (node input is not needed as BSM is continuous)

#test set 2
#23.163638908880593 (binomial)
#23.167471170306726 (BSM)


#%%

#--------------------------------------------------------------------------------
"""
run (%matplotlib qt)  in the console to allow for an interactive plot, remove the parentheses

this allows the figures to pop up on a separate window 
the different values in each point can be seen when hovering over the table with a cursor
"""
#%%

""" 

PLOTTING BINOMIAL PRICE AGAINST BSM PRICE

This section plots the price generated by the Binomial Tree against 
the price generated by the Black-Scholes-Merton Model

"""

#(I)
#setting node as range to loop over 
node_range = range(1, 501) 

#setting object to store results
BSMvalues = []
Binomvalues = []

#iteration
for n_nodes in node_range:
    Binom = binomialpricing(S0, K, T, r, sigma, n_nodes, setting)
    BSM = BSM_price(S0, K, T, r, sigma, setting)
    BSMvalues.append(BSM)
    Binomvalues.append(Binom)

plt.clf() #clear previous plot

#plot Binomial price(x-value, y-value)
plt.plot(node_range, Binomvalues, label="Binomial Price")

#plot BSM price (x-value, y-value)
plt.plot(node_range, BSMvalues, label="BSM Price")

#axis labels
plt.xlabel("Number of Steps")
plt.ylabel("Option Price")

#legend
plt.legend()

#add grid
plt.grid(True)
plt.show()

"""
The resulting plot should show that the price from the Binomial tree converges to the BSM price as the node increases

this means that as the number of nodes approaches infinity the Binomial model will approach the BSM price

this is because as the number of nodes increase, the discreet time value approaches the continuous 
value used in the BSM model

This allows us to verify that the binomial pricing model is accurate 

However, as the Binomial Pricing model incorporates the expected value at all time steps 
this method is typically used to price American options that allows for early exercise
"""

#%%

"""
PLOTTING THE VALUE OF AN OPTION AGAINST TIME
"""
#(I)
##Option Value as it approaches maturity

#making K the same as S0 for simplicity of visualisation to compare between call and put
Kc = S0   # strike

# Time to expiry from 1 year → 1 day
Tc = np.linspace(1, 0.003, 200)

# Compute option values
put_values = [BSM_price(S0, Kc, t, r, sigma, option_type='put') for t in Tc]
call_values = [BSM_price(S0, Kc, t, r, sigma, option_type='call') for t in Tc]

#checking variables are 1 dimensional, output should be (200,)

print(Tc.shape)
print(np.array(call_values).shape)
print(np.array(put_values).shape)

#%%

# Plot
plt.clf()

#plotting put and call values
plt.plot(Tc, put_values, label = "Put Value")
plt.plot(Tc, call_values, label = "Call Value")

#axis labels
plt.xlabel("Time to Expiry (Years)")
plt.ylabel("Option Value")

#legend
plt.legend()
plt.title("Time Decay of an Option")

#shows time flowing left→right (expiry)
plt.gca().invert_xaxis()  

#add grid
plt.grid(True)
plt.savefig("put_call_time_decay.png")
plt.show()

"""
Through the plot we can observe that the value of a call and a put decays as it approaches maturity
"""

#%%

#checking the values are consistent
print(S0,'\n',K,'\n',T,'\n',r,'\n',sigma,'\n',node)

#%%
"""
#PLOTTING OPTION VALUES AGAINST S
"""
#%%
#(I)
# Range of underlying prices 
S_values = np.linspace(0.8*S0, 1.2*S0, 200)

# Compute option prices
call_values = BSM_price(S_values, K, T, r, sigma, option_type='call')
put_values  = BSM_price(S_values, K, T, r, sigma, option_type='put')


#checking variables are 1 dimensional, output should be (200,)

print(S_values.shape)
print(np.array(call_values).shape)
print(np.array(put_values).shape)
#%%

# plotting a new figure
plt.figure(figsize=(10,6))  # plot new figure
plt.plot(S_values, call_values, label="Call Value", color='blue')
plt.plot(S_values, put_values, label="Put Value", color='red')
plt.legend()
plt.xlabel("Stock Price (S)")
plt.ylabel("Option Value")
plt.title("Stock Price vs. European Option Value")
plt.grid(True)
plt.savefig("call_vs_stock.png")
plt.show()
"""
From the plot we can observe that the value of a call and a put move in opposite direction as S changes

"""


#%%
"""
GREEKS

Greeks are a set of measures that describe how the price of an option changes against various factors

"""

#%%

#checking the values are consistent
print(S0,'\n',K,'\n',T,'\n',r,'\n',sigma,'\n',node)

#%%

#%%

#(II)
#code adapted from codearmo explanation how to calculate greeks
#https://www.codearmo.com/python-tutorial/options-trading-greeks-black-scholes
#Creating a function that calculates the greeks

def BSM_greeks(S0, K, r, sigma, T, option_type="put"):
    """
   Calculates the Black-Scholes-Merton (BSM) option Greeks for a European option

   Parameters:
   -----------
   S0 : float or array
       Current price of the underlying asset.
   K : float
       Strike price of the option.
   r : float
       Risk-free interest rate (annualized).
   sigma : float
       Volatility of the underlying asset (annualized).
   T : float
       Time to expiration in years.
   option_type : str
       Type of option, either 'call' or 'put'

   Returns:
   --------
   dict
       Dictionary containing the five main option Greeks:
       - delta : Sensitivity of option price to underlying asset price.
       - gamma : Sensitivity of delta to underlying asset price.
       - vega  : Sensitivity of option price to volatility.
       - theta : Sensitivity of option price to time decay.
       - rho   : Sensitivity of option price to interest rate.
   
   Notes:
   ------
   - Handles scalar or array inputs for S0.
   - Ensures no division by zero for price, volatility, or time.
   - Raises ValueError if option_type is not 'call' or 'put'.
   """
    #converts S0 into an array of stock prices
    S = np.array(S0, dtype=float)
    S = np.where(S <= 0, 1e-8, S)  # avoid zero or negative underlying prices
    sigma = max(sigma, 1e-8)
    T = max(T, 1e-6)

    # Z-scores
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    #converts any input into lower case
    option_type = option_type.lower()

    if option_type == "call":
        #calculate delta, theta, rho for calls
        
        delta = norm.cdf(d1)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) \
                - r * K * np.exp(-r*T) * norm.cdf(d2)
        rho   = K * T * np.exp(-r*T) * norm.cdf(d2)

    elif option_type == "put":
        #calculate delta, theta, rho for puts
        
        delta = norm.cdf(d1) - 1
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) \
                + r * K * np.exp(-r*T) * norm.cdf(-d2)
        rho   = -K * T * np.exp(-r*T) * norm.cdf(-d2)
    
    #executed if input is not expected must be call or put
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    #calculate gamma and vega
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    
    #output is a dictionary
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho
    }

#calculates greeks on the initial parameters

greeks = BSM_greeks(S0,K,r,sigma,T, setting)

delta = greeks["delta"]
gamma = greeks["gamma"]
vega  = greeks["vega"]
theta = greeks["theta"]
rho   = greeks["rho"]

print(delta, gamma, vega, theta, rho)


#%%

#PLOTTING GREEKS

#%%

#getting real world options data
#(II) video to learn what keys needed to retrieve the data
#https://www.youtube.com/watch?v=ZLbVsPy13QI
ticker = yf.Ticker(tckr)  # tckr is your ticker symbol, e.g., 'AAPL'
expiry = ticker.options[0]  # take the first available expiry
option_chain = ticker.option_chain(expiry)
calls = option_chain.calls
puts = option_chain.puts

S0_market = ticker.history(period="1d")['Close'].iloc[-1]

#get expiry date of available option
expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
T_market = (expiry_date - datetime.today()).days / 365
r_market = r  # default risk-free rate
#this risk free rate will be the one that plots the greeks, 
#you can change it on the plot or adjust it before executing the plot 

# Function to get ATM strike and implied volatility
def get_option_data(option_type="call"):
    chain = calls if option_type=="call" else puts
    K = chain['strike'].iloc[(chain['strike'] - S0_market).abs().argmin()]
    sigma = chain.loc[chain['strike']==K, 'impliedVolatility'].values[0]
    return K, sigma

# Set initial option parameters
option_type_init = setting # default, can be 'put'
K_init, sigma_init = get_option_data(option_type_init)
S0_init = S0_market
r_init = r_market
T_init = T_market

#if T_init is 0 then we use the initial input of T as the value
if T_init == 0:
    T_init = T
    print(f'T_init was zero (invalid) default T value was used:{T}')

#check for any 0 values
print(S0_init, K_init, r_init, T_init, option_type_init)
#%%


"""
#INTERACTIVE PLOT
"""

"""
This block creates 2 interactive figures based off the BSM_price function and the BSM_greeks

it includes textboxes to allow users to adjust:
    • S0 (current underlying price)
    • K  (strike price)
    • r  (risk-free rate)
    • σ  (volatility)
    • T  (time to maturity)
    • option type : call/put
    
figure will show the greeks plot that will update with the input from the textboxes
the other will provide the plot of the option Value 

run %matplotlib qt in console before executing this code to allow for exploration and adjusting variables

"""


#Greeks plotting

fig, axes = plt.subplots(5, 1, figsize=(10, 12))
plt.subplots_adjust(left=0.1, bottom=0.25, hspace=1.5, top=0.95)
titles = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
lines = []

S_range = np.linspace(0.5*K_init, 1.5*K_init, 200)
greeks = BSM_greeks(S_range, K_init, r_init, sigma_init, T_init, option_type_init)
greeks_data = [greeks["delta"], greeks["gamma"], greeks["vega"], greeks["theta"], greeks["rho"]]

#(II) help from chatgpt here to figure out how to loop through more than 3 inputs for multiple objects 
#this line iterates over each subplot axis objects 
for ax, data, title in zip(axes, greeks_data, titles):
    
    #plots the greeks against S_range
    line, = ax.plot(S_range, data, color='blue')  # Greek curves always blue
    s0_line = ax.axvline(S0_init, color='blue', linestyle='--')  # vertical line color changes
    label_text = ax.text(0.95, 0.85, '', transform=ax.transAxes, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.8))
    ax.set_title(title)
    ax.set_xlabel("Underlying Price S")
    ax.set_ylabel(title)
    ax.grid(True)
    lines.append((line, s0_line, label_text))
#(I)
#https://shreyaskhatri.medium.com/using-widgets-to-make-plotting-using-matplotlib-easy-using-football-data-518e514feff6
#referred to website for guidance on plotting syntax
# widgets----------------------------------------------------------------------
axcolor = 'lightgoldenrodyellow'
ax_S0 = plt.axes([0.10, 0.12, 0.2, 0.045], facecolor=axcolor)
ax_K = plt.axes([0.35, 0.12, 0.2, 0.045], facecolor=axcolor)
ax_r = plt.axes([0.10, 0.05, 0.2, 0.045], facecolor=axcolor)
ax_sigma = plt.axes([0.35, 0.05, 0.2, 0.045], facecolor=axcolor)
ax_T = plt.axes([0.60, 0.05, 0.2, 0.045], facecolor=axcolor)

text_S0 = TextBox(ax_S0, 'S0', initial=str(S0_init))
text_K = TextBox(ax_K, 'K', initial=str(K_init))
text_r = TextBox(ax_r, 'r', initial=str(r_init))
text_sigma = TextBox(ax_sigma, 'σ', initial=str(sigma_init))
text_T = TextBox(ax_T, 'T', initial=str(T_init))

ax_option = plt.axes([0.60, 0.12, 0.15, 0.08], facecolor=axcolor)
radio_option = RadioButtons(ax_option, ['call', 'put'], active=0 if option_type_init=="call" else 1)

#Option Price plotting---------------------------------------------------------
fig_price, ax_price = plt.subplots(figsize=(8, 4))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
option_values = BSM_price(S_range, K_init, r_init, sigma_init, T_init, option_type_init)
price_line, = ax_price.plot(S_range, option_values, color='purple', label='Option Price')
s0_line_price = ax_price.axvline(S0_init, color='blue', linestyle='--')
status_text_price = ax_price.text(0.95, 0.85, '', transform=ax_price.transAxes, fontsize=10,
                                  bbox=dict(facecolor='white', alpha=0.8))
ax_price.set_title("Option Price")
ax_price.set_xlabel("Underlying Price S")
ax_price.set_ylabel("Price")
ax_price.grid(True)
ax_price.legend()

# Update function--------------------------------------------------------------
#(I)
#this function adjusts the values in the plot automatically after the inputs are changed in the textbox
def update_plot(event=None):
    """ 
    This function allows for the plots to be interactive by reading the input 
    in the textbox and adjusting the plots based on those values
    """
    try:
        S0 = max(float(text_S0.text), 1e-8)
        K = max(float(text_K.text), 1e-8)
        r = float(text_r.text)
        sigma = max(float(text_sigma.text), 1e-8)
        T = max(float(text_T.text), 1e-6)
        option_type = radio_option.value_selected

        S_range_new = np.linspace(0.5*K, 1.5*K, 200)

        # Update Greeks
        greeks = BSM_greeks(S_range_new, K, r, sigma, T, option_type)
        greeks_data = [greeks["delta"], greeks["gamma"], greeks["vega"], greeks["theta"], greeks["rho"]]

        # Update Option Price
        option_values = BSM_price(S_range_new, K, r, sigma, T, option_type)

        # Determine ATM/ITM/OTM
        if abs(S0-K)/K < 0.01:
            status = "ATM"
            color = 'blue'
        elif (option_type=="call" and S0>K) or (option_type=="put" and S0<K):
            status = "ITM"
            color = 'green'
        else:
            status = "OTM"
            color = 'red'

        # Update Greek plots
        for (line, s0_line, label_text), data in zip(lines, greeks_data):
            line.set_ydata(data)
            s0_line.set_xdata([S0, S0])
            s0_line.set_color(color)
            label_text.set_text(status)
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

        # Update Option Price plot
        price_line.set_ydata(option_values)
        s0_line_price.set_xdata([S0, S0])
        s0_line_price.set_color(color)
        status_text_price.set_text(status)
        ax_price.relim()
        ax_price.autoscale_view()
        fig_price.canvas.draw_idle()

    except ValueError:
        pass

# ---------- Connect widgets ----------
text_S0.on_submit(update_plot)
text_K.on_submit(update_plot)
text_r.on_submit(update_plot)
text_sigma.on_submit(update_plot)
text_T.on_submit(update_plot)
radio_option.on_clicked(update_plot)

plt.show()


#%%

#PLOTTING DIFFERENT OPTION POSITIONS 

#(I)
# this function dictates how many values of S is computed in each of the payoffs below
def get_S_range(S0, width=0.9, points=500):
    '''
    Creates a dynamic S-range based on S0.
    width = percentage above/below S0 (0.9 = ±20%)
    points = number of price steps
    
    this will be used to calculate the payoffs
    
    '''
    low = (1 - width) * S0
    high = (1 + width) * S0
    return np.linspace(low, high, points)

# Single underlying price array S_range

# PAYOFF FUNCTIONS
'''
Calculates and plots the payoffs of the 4 basic positions in options

long call : Buying the right to buy
short call : Selling the right to buy
long put : Buying the right to sell
short put : Selling the right to sell
'''

def long_call_payoff(S0, K, r, sigma, T):
    S_range = get_S_range(S0)
    premium = BSM_price(S0, K, r, sigma, T, "call")
    payoff = np.maximum(S_range - K, 0) - premium
    return S_range, payoff

def short_call_payoff(S0, K, r, sigma, T):
    S_range = get_S_range(S0)
    premium = BSM_price(S0, K, r, sigma, T, "call")
    payoff = -np.maximum(S_range - K, 0) + premium
    return S_range, payoff

def long_put_payoff(S0, K, r, sigma, T):
    S_range = get_S_range(S0)
    premium = BSM_price(S0, K, r, sigma, T, "put")
    payoff = np.maximum(K - S_range, 0) - premium
    return S_range, payoff

def short_put_payoff(S0, K, r, sigma, T):
    S_range = get_S_range(S0)
    premium = BSM_price(S0, K, r, sigma, T, "put")
    payoff = -np.maximum(K - S_range, 0) + premium
    return S_range, payoff

def plot_basic_option_payoffs(S0, K, r, sigma, T):
    # Compute payoffs
    S_range_call_long, payoff_call_long = long_call_payoff(S0, K, r, sigma, T)
    S_range_call_short, payoff_call_short = short_call_payoff(S0, K, r, sigma, T)
    S_range_put_long, payoff_put_long = long_put_payoff(S0, K, r, sigma, T)
    S_range_put_short, payoff_put_short = short_put_payoff(S0, K, r, sigma, T)

    # Create a 2×2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # List of plots
    plots = [
        ("Long Call", S_range_call_long, payoff_call_long),
        ("Short Call", S_range_call_short, payoff_call_short),
        ("Long Put", S_range_put_long, payoff_put_long),
        ("Short Put", S_range_put_short, payoff_put_short)
    ]

    for ax, (title, S_range, payoff) in zip(axes.flatten(), plots):
        ax.plot(S_range, payoff, linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(S0, color="gray", linestyle="--", label=f"S0 = {S0}")
        ax.set_title(title)
        ax.set_xlabel("Underlying Price (S)")
        ax.set_ylabel("Payoff")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
plot_basic_option_payoffs(S0, K, r, sigma, T)

#%%
#STRATEGY PAYOFFS FUNCTIONS

#(I)
"""
The functions below include 8 different strategies that can be used when taking a position in options

1. Covered call         : Long asset, Short call
2. Protective put       : Long asset, Long put
3. Bull spread (call)   : Long call at K1, Short call at K2
4. Bear spread (call)   : Short call at K1, Long call at K2
5. straddle             : Long call, Long Put at strike K
6. strangle             : Long put at K1, Long call at K2
7. Butterfly spread     : Long Call at K1, 2 Short calls at K2, 1 Long call at K3
8. Iron condor          : Long put at K1, Short put at K2, Short call at K3, Long call at K4

where K1<K2<K3<K4


"""
def covered_call(current_price, K, r, sigma, T):
    S_range = get_S_range(current_price)
    premium = BSM_price(current_price, K, r, sigma, T, "call")
    return S_range, (S_range - current_price) + (-np.maximum(S_range - K, 0) + premium)
     #(S_range - current_price) is the long underlying position
     #(-np.maximum(S_range - K, 0) + premium) the short call payoff
     
def protective_put(current_price, K, r, sigma, T):
    S_range = get_S_range(current_price)
    premium = BSM_price(current_price, K, r, sigma, T, "put")
    return S_range, (S_range - current_price) + (np.maximum(K - S_range, 0) - premium)
    #(S_range - current_price) is the long underlying position
    #(np.maximum(K - S_range, 0) - premium)long put position
    
    
def bull_call_spread(current_price, K1, K2, r, sigma, T):
    S_range = get_S_range(current_price)
    premium1 = BSM_price(current_price, K1, r, sigma, T, "call")
    premium2 = BSM_price(current_price, K2, r, sigma, T, "call")
    return S_range, (np.maximum(S_range - K1, 0) - premium1) + (-np.maximum(S_range - K2, 0) + premium2)
    #(np.maximum(S_range - K1, 0) - premium1) buying lower strike call
    #(-np.maximum(S_range - K2, 0) + premium2) selling higher strike call

def bear_call_spread(current_price, K1, K2, r, sigma, T):
    S_range = get_S_range(current_price)
    premium1 = BSM_price(current_price, K1, r, sigma, T, "call")
    premium2 = BSM_price(current_price, K2, r, sigma, T, "call")
    return S_range, (-np.maximum(S_range - K1, 0) + premium1) + (np.maximum(S_range - K2, 0) - premium2)
    #(-np.maximum(S_range - K1, 0) + premium1) sell lower strike call
    #(np.maximum(S_range - K2, 0) - premium2) buy higher strike call
    
def straddle(current_price, K, r, sigma, T):
    S_range = get_S_range(current_price)
    premium_call = BSM_price(current_price, K, r, sigma, T, "call")
    premium_put = BSM_price(current_price, K, r, sigma, T, "put")
    return S_range, (np.maximum(S_range - K, 0) - premium_call) + (np.maximum(K - S_range, 0) - premium_put)
    #buy call + put at the same strike
    
def strangle(current_price, K1, K2, r, sigma, T):
    if not K1 < K2 :
        raise ValueError("Strikes must satisfy K1 < K2")
    S_range = get_S_range(current_price)
    premium_call = BSM_price(current_price, K2, r, sigma, T, "call")
    premium_put = BSM_price(current_price, K1, r, sigma, T, "put")
    return S_range, (np.maximum(S_range - K2, 0) - premium_call) + (np.maximum(K1 - S_range, 0) - premium_put)
    #buy call at higher strike
    #buy put at lower strike    


def butterfly_spread(current_price, K1, K2, K3, r, sigma, T):
    """
    Butterfly spread: long K1 call, short 2 K2 calls, long K3 call
    Assumes K1 < K2 < K3
    """
    if not K1 < K2 < K3:
        raise ValueError("Strikes must satisfy K1 <= K2 <= K3")
    S_range = get_S_range(current_price)
    
    premium1 = BSM_price(current_price, K1, r, sigma, T, "call")
    premium2 = BSM_price(current_price, K2, r, sigma, T, "call")
    premium3 = BSM_price(current_price, K3, r, sigma, T, "call")
    
    payoff = (np.maximum(S_range - K1, 0) - premium1) \
             + (-2*np.maximum(S_range - K2, 0) + 2*premium2) \
             + (np.maximum(S_range - K3, 0) - premium3)
    
    return S_range, payoff


def iron_condor(current_price, K1, K2, K3, K4, r, sigma, T, include_premium=True):
    """
    Iron Condor: short put K1, long put K2, short call K3, long call K4
    Assumes K1 < K2 < K3 < K4
    Returns: S_range, payoff
    """
    if not K1 < K2 < K3 < K4:
        raise ValueError("Strikes must satisfy K1 < K2 < K3 < K4")
    
    # Generate a range of underlying prices
    S_range = np.linspace(K1 - (K4-K1)*0.2, K4 + (K4-K1)*0.2, 1000)
    
    # Intrinsic payoff at expiry
    short_put  = -np.maximum(K2 - S_range, 0)
    long_put   = np.maximum(K1 - S_range, 0)
    short_call = -np.maximum(S_range - K3, 0)
    long_call  = np.maximum(S_range - K4, 0)
    
    payoff = short_put + long_put + short_call + long_call
    
    if include_premium:
        # Approximate option premiums using BSM
        premium1 = BSM_price(current_price, K1, r, sigma, T, "put")
        premium2 = BSM_price(current_price, K2, r, sigma, T, "put")
        premium3 = BSM_price(current_price, K3, r, sigma, T, "call")
        premium4 = BSM_price(current_price, K4, r, sigma, T, "call")
        
        # Net premium received from the iron condor
        net_premium = premium1 - premium2 + premium3 - premium4
        payoff += net_premium
    
    return S_range, payoff



#%%
"""
#PLOTTING ALL STRATEGIES
#you can observe the different strategies by executing this block of code

"""
#%%
'''
This section produces a pop up of the plots of the 8 different strategies payoffs
The specifications are defaulted to the below variables.
'''
#(I)
# parameters
current_price = S0

# Strike selections for different strategies
K = S0
K1, K2 = S0-100, S0+100
# for butterfly spread
Kb1, Kb2, Kb3 = S0-10, S0, S0+10
# for iron condor
Kc1, Kc2, Kc3, Kc4 = S0-100, S0-70, S0+70, S0+100

strategies = [
    ("Covered Call", covered_call, (current_price, K, r, sigma, T)),
    ("Protective Put", protective_put, (current_price, K, r, sigma, T)),
    ("Bull Call Spread", bull_call_spread, (current_price, K1, K2, r, sigma, T)),
    ("Bear Call Spread", bear_call_spread, (current_price, K1, K2, r, sigma, T)),
    ("Straddle", straddle, (current_price, K, r, sigma, T)),
    ("Strangle", strangle, (current_price, K1, K2, r, sigma, T)),
    ("Butterfly Spread", butterfly_spread, (current_price, Kb1, Kb2, Kb3, r, sigma, T)),
    ("Iron Condor", iron_condor, (current_price, Kc1, Kc2, Kc3, Kc4, r, sigma, T))
]

# Figure setup
n_strategies = len(strategies)
n_cols = 3
n_rows = (n_strategies + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*4))
axes = axes.flatten()

for ax, (title, func, params) in zip(axes, strategies):
    S_range, payoff = func(*params)
    ax.plot(S_range, payoff, linewidth=2, label='Payoff')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(current_price, color='gray', linestyle='--', label=f"Current Price = {current_price}")
    ax.set_title(title)
    ax.set_xlabel("Underlying Price (S)")
    ax.set_ylabel("Profit / Loss")
    ax.grid(True)
    
    # Optional: auto-scale y-axis to max loss/profit
    buffer = (max(payoff) - min(payoff)) * 0.1
    ax.set_ylim(min(payoff)-buffer, max(payoff)+buffer)
    
    ax.legend()

# Remove unused axes
for ax in axes[len(strategies):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()


#%%

#checking the variables are consistent
print(S0,'\n',K,'\n',T,'\n',r,'\n',sigma,'\n',node)

#%%
"""
THIS PLOT IS TO SHOW THAT THE VALUE OF THE PORTFOLIO WILL CONVERGE WITH THE PAYOFF AS IT APPROACHES MATURITY

this example uses the payoff of the Bull call spread
"""
#%%
"""
Adjust Parameters for the Spread here

"""
current_price = S0_market
K1, K2 = S0_market-20, S0_market+20 #K1&K2 has been simplified for the development process
T_total = T

def bull_call_portfolio_value(current_price, K1, K2, r, sigma, T_total, t):
    S_range = get_S_range(current_price)
    T_remaining = max(T_total - t, 0)
    premium1 = BSM_price(current_price, K1, r, sigma, T_total, "call")
    premium2 = BSM_price(current_price, K2, r, sigma, T_total, "call")
    # Portfolio value uses option values at T_remaining
    # T remaining decreases as t increases, help from chatgpt for this function to plot portfolio value
    value = (BSM_price(S_range, K1, r, sigma, T_remaining, "call") - premium1) + \
            (-BSM_price(S_range, K2, r, sigma, T_remaining, "call") + premium2)
    return S_range, value

# ------------------ Parameters ------------------

# Compute payoff at expiration
S_range_exp, payoff_exp = bull_call_spread(current_price, K1, K2, r, sigma, T_total)
S_range_val, portfolio_val = bull_call_portfolio_value(current_price, K1, K2, r, sigma, T_total, t=0.0)

# ------------------ Plot ------------------
fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(bottom=0.25)

line_payoff, = ax.plot(S_range_exp, payoff_exp, label='Payoff at Expiration', color='blue', linewidth=2)
line_portfolio, = ax.plot(S_range_val, portfolio_val, linestyle='--', label='Portfolio Value', color='red', linewidth=2)
ax.axvline(current_price, linestyle='--', color='gray', label=f'Current Price = {current_price}')
ax.set_xlabel('Underlying Price')
ax.set_ylabel('Profit / Loss')
ax.set_title('Bull Call Spread: Payoff vs Portfolio Value Over Time')
ax.grid(True)
ax.legend()
#lala

# ------------------ Slider for time ------------------
ax_slider_time = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_time = Slider(ax_slider_time, 't', 0.0, T_total, valinit=0.0)

def update(t):
    _, portfolio_val = bull_call_portfolio_value(current_price, K1, K2, r, sigma, T_total, t)
    line_portfolio.set_ydata(portfolio_val)
    fig.canvas.draw_idle()

slider_time.on_changed(update)
plt.show()



#%%
"""
#FUNCTION TO RUN THROUGH CHOSEN STRATEGY

"""

#(II)
#help form chatgpt here to figure out how to make a function that can go through different functions based on a selected strategy on the input
def pickstrat_plot(strategy, strikes, K_names, current_price=None, r=0.05, sigma=0.2, T=1.0):
    """
    Plots the payoff diagram of a given options strategy with interactive textboxes for strike prices.

    Parameters:
    -----------
    strategy : function
        The options strategy function to plot (e.g., covered_call, protective_put).
        Must return a tuple (S_range, payoff_values).
    strikes : tuple or list
        Strike prices used in the strategy. Length depends on the strategy.
    K_names : list of str
        Names for each strike, displayed in textboxes for interactive editing.
    current_price : float, optional
        Current price of the underlying asset. Defaults to the first strike if not provided.
    r : float, optional
        Risk-free interest rate (default 0.05).
    sigma : float, optional
        Volatility of the underlying asset (default 0.2).
    T : float, optional
        Time to expiration in years (default 1.0).

    Functionality:
    --------------
    - Generates a price range around the current underlying price.
    - Computes the payoff of the strategy for all prices in the range.
    - Creates a plot of Profit/Loss vs. Underlying Price.
    - Adds textboxes for each strike, allowing users to dynamically update strikes.
    - Updates the payoff curve and rescales axes when any textbox value changes.
    - Highlights the current underlying price with a vertical dashed line.
    """
    # Ensure strikes is a tuple
    strikes = tuple(strikes)

    # If current_price not provided, default to first strike
    if current_price is None:
        current_price = strikes[0]
    
    #if the strategy is for the iron condor use the S_range generated by the function
    if strategy.__name__ == "iron_condor":# help here from chatgpt to figure out how to refer to strategy name based on calling a function
        y = strategy(*((current_price,) + strikes + (r, sigma, T)))
        S_range = y[0]  # S_range comes from strategy
        payoff = y[1]
    #if it is any other strategy we can use the S_range from the get_S_range function
    else:
        # For all other strategies, generate S_range normally
        S_range = get_S_range(current_price)
        y = strategy(*((current_price,) + strikes + (r, sigma, T)))
        payoff = y[1]
    
    #create the figure 
    fig, axis = plt.subplots(figsize =(10,12))
    plt.subplots_adjust(bottom=0.35)
    
    #plot initial payoff
    line, = axis.plot(S_range, payoff, color = "blue", linewidth = 2)
    axis.axvline(current_price, linestyle = '--', color='gray', label=f'Current Price = {current_price}')
    axis.axhline(0, linewidth=1)
    axis.set_xlabel('Underlying Price')
    axis.set_ylabel('Profit/Loss')
    axis.set_title(f'{strategy.__name__} Payoff')
    axis.grid(True)
    axis.legend()
    
    # Create textboxes for each strike parameter
    text_boxes = []
    for i, name in enumerate(K_names):
        axbox = plt.axes([0.15, 0.25 - i*0.05, 0.2, 0.03], facecolor="lightgoldenrodyellow")
        text_box = TextBox(axbox, name, initial=str(strikes[i]))
        text_boxes.append(text_box)

    # Update function when user edits parameters
    def submit(_):
        try:
            # Convert inputs to floats
            new_strikes = tuple(float(tb.text) for tb in text_boxes)

            # Some strategies require current_price to update as well (optional)
            updated_current = new_strikes[0] if strategy.__name__ in [
                'covered_call', 'protective_put', 'straddle'
            ] else current_price

            # Recompute payoff
            y_new = strategy(*((updated_current,) + new_strikes + (r, sigma, T)))

            # # Update line on plot
            line.set_ydata(y_new[1])
            line.set_xdata(y_new[0])

            # Dynamically rescale axes with a small margin
            margin_y = 0.05 * (max(y_new[1]) - min(y_new[1]))
            axis.set_xlim(min(y_new[0]), max(y_new[0]))
            axis.set_ylim(min(y_new[1]) - margin_y, max(y_new[1]) + margin_y)

            fig.canvas.draw_idle()

        except ValueError:
            print("Please enter valid numbers in the text boxes.")

    # Link all textboxes to the update function
    for tb in text_boxes:
        tb.on_submit(submit)

    plt.show()
    
    
#%%

"""

PLOT YOUR CHOICES OF STRATEGY AND K VALUES

to adjust the initial plot change the values of the K for the item strikes

You can test different values of K once the plot is generated by entering through the textboxes

current_price, r, sigma and T are defaulted at the original inputs for each variable

"""
print(current_price,'\n',K,'\n',T,'\n',r,'\n',sigma,'\n',node)

"""
Adjust input values here 
"""
current_price = S0
r = r
sigma = sigma
T = T

#%%
#COVERED CALL
#%%
pickstrat_plot(
    strategy=covered_call,
    strikes=[K],
    K_names=["K"],
    current_price=current_price,
    r=r, sigma=sigma, T=T
)
#%%
#PROTECTIVE PUT
#%%
pickstrat_plot(
    strategy=protective_put,
    strikes=[K],
    K_names=["K"],
    current_price=current_price,
    r=r, sigma=sigma, T=T
)


#%%
#BULL SPREAD (CALL)
#%%
pickstrat_plot(
    strategy=bull_call_spread,
    strikes=[K1, K2],
    K_names=["K1", "K2"],
    current_price=current_price,
    r=r, sigma=sigma, T=T
)

#%%
#BEAR SPREAD (CALL)
#%%
pickstrat_plot(
    strategy=bear_call_spread,
    strikes=[K1, K2],
    K_names=["K1", "K2"],
    current_price=current_price,
    r=r, sigma=sigma, T=T
)

#%%
#STRADDLE
#%%
pickstrat_plot(
    strategy=straddle,
    strikes=[K],
    K_names=["K"],
    current_price=current_price,
    r=r, sigma=sigma, T=T
)

#%%
#STRANGLE
#%%
pickstrat_plot(
    strategy=strangle,
    strikes=[K1, K2],
    K_names=["K1", "K2"],
    current_price=current_price,
    r=r, sigma=sigma, T=T
)


#%%
#BUTTERFLY SPREAD
#%%
pickstrat_plot(
    strategy=butterfly_spread,
    strikes=[Kb1, Kb2, Kb3],
    K_names=["K1", "K2", "K3"],
    current_price=current_price,
    r=r, sigma=sigma, T=T
)


#%%
#IRON CONDOR
#%%
pickstrat_plot(
    strategy=iron_condor,
    strikes=[Kc1, Kc2, Kc3, Kc4],
    K_names=["K1", "K2", "K3", "K4"],
    current_price=current_price,
    r=r, sigma=sigma, T=T
)


#%%

def compute_payoff(strategy, current_price, strikes, r=0.01, sigma=0.2, T=0.5, include_premium=True):
    """
    Calls the appropriate strategy function based on the strategy name.
    
    strategy: string, e.g., "covered_call", "bull_call_spread"
    strikes: list of strike prices needed for the strategy
    """
    if strategy == "covered_call":
        return covered_call(current_price, strikes[0], r, sigma, T)
    elif strategy == "protective_put":
        return protective_put(current_price, strikes[0], r, sigma, T)
    elif strategy == "bull_call_spread":
        return bull_call_spread(current_price, strikes[0], strikes[1], r, sigma, T)
    elif strategy == "bear_call_spread":
        return bear_call_spread(current_price, strikes[0], strikes[1], r, sigma, T)
    elif strategy == "straddle":
        return straddle(current_price, strikes[0], r, sigma, T)
    elif strategy == "strangle":
        return strangle(current_price, strikes[0], strikes[1], r, sigma, T)
    elif strategy == "butterfly_spread":
        return butterfly_spread(current_price, strikes[0], strikes[1], strikes[2], r, sigma, T)
    elif strategy == "iron_condor":
        return iron_condor(current_price, strikes[0], strikes[1], strikes[2], strikes[3], r, sigma, T, include_premium)
    else:
        raise ValueError("Strategy not recognized")



def plot_strategy_payoff(strategy, current_price, strikes, r=0.01, sigma=0.2, T=0.5, include_premium=True):
    """
    Plot payoff for any strategy.
    
    strategy: string, name of strategy function
    strikes: list of strike prices (length depends on strategy)
    """
    S_range, payoff = compute_payoff(strategy, current_price, strikes, r, sigma, T, include_premium)
    
    plt.figure(figsize=(10,6))
    plt.plot(S_range, payoff, label=f"{strategy} payoff")
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Underlying Price")
    plt.ylabel("Payoff")
    plt.title(f"{strategy} Payoff Diagram")
    plt.grid(True)
    plt.legend()
    plt.show()
    

# Example usage:
plot_strategy_payoff(
    strategy="bull_call_spread",
    current_price=100,
    strikes=[95, 105],
    r=0.01,
    sigma=0.2,
    T=0.5
)

#%%


# Stock price range
S_values = np.linspace(0.8*K1, 1.2*K2, 200)

# Compute bull call spread payoff/value
S_range, bull_spread_value = bull_call_spread(S_values, K1, K2, r, sigma, T)  # make sure it returns (S_range, value)
ybull = bull_spread_value  # should be 1D array

# Compute call and put values for comparison
call_values = BSM_price(S_values, K, r, sigma, T, option_type='call')
put_values  = BSM_price(S_values, K, r, sigma, T, option_type='put')

# Check shapes
print(S_values.shape)  # (200,)
print(ybull.shape)     # (200,)
print(call_values.shape)
print(put_values.shape)

# Plot
plt.figure(figsize=(10,6))
plt.plot(S_values, ybull, label="Bull Call Spread Value", color='blue')
plt.plot(S_values, call_values, label="Call Value", color='green', linestyle='--')
plt.plot(S_values, put_values, label="Put Value", color='red', linestyle='--')
plt.xlabel("Stock Price (S)")
plt.ylabel("Option Value / Payoff")
plt.title("Option Values vs. Stock Price")
plt.grid(True)
plt.legend()
plt.savefig("call_vs_stock.png")
plt.show()

#%%
S_range, bull_spread_value = bull_call_spread(S_values, K1, K2, r, sigma, T)
print(S_range.shape)
print(bull_spread_value.shape)
print(bull_spread_value[:10])  # print first 10 values

print("S_values:", S_values[:5])
print("ybull:", ybull[:5])



#%%
def portfolio_value_at_current(strategy_name, current_price, strikes, r, sigma, T_total, t):
    """
    Compute the portfolio value at time t for the current underlying price.
    """
    T_remaining = max(T_total - t, 0)
    
    # Handle each strategy dynamically
    if strategy_name == "covered_call":
        premium = BSM_price(current_price, strikes[0], r, sigma, T_total, "call")
        value = (current_price - current_price) + (-max(current_price - strikes[0], 0) + premium)
        
    elif strategy_name == "protective_put":
        premium = BSM_price(current_price, strikes[0], r, sigma, T_total, "put")
        value = (current_price - current_price) + (max(strikes[0] - current_price, 0) - premium)
        
    elif strategy_name == "bull_call_spread":
        premium1 = BSM_price(current_price, strikes[0], r, sigma, T_total, "call")
        premium2 = BSM_price(current_price, strikes[1], r, sigma, T_total, "call")
        value = (max(current_price - strikes[0], 0) - premium1) + (-max(current_price - strikes[1], 0) + premium2)
    
    elif strategy_name == "iron_condor":
        # Use your existing iron_condor logic but only for current_price
        S_range, payoff = iron_condor(current_price, *strikes, r, sigma, T_remaining)
        # Find the index corresponding to current_price
        idx = np.argmin(np.abs(S_range - current_price))
        value = payoff[idx]
    
    else:
        raise ValueError(f"Strategy {strategy_name} not implemented yet.")
    
    return value

#%%

strategy_choice = "bull_call_spread"
strikes = [S0_market-20, S0_market+20]
t_current = 0.5  # Halfway to maturity, for example

portfolio_val_now = portfolio_value_at_current(strategy_choice, S0_market, strikes, r, sigma, T, t_current)
print(f"Portfolio value at current price {S0_market} and t={t_current}: {portfolio_val_now}")


#%%

# making a universal payoff calculator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider

def plot_strategy_and_portfolio(strategy_func, strikes, K_names, current_price, r=0.05, sigma=0.2, T_total=1.0):
    """
    Plots both the payoff at expiration and the portfolio value at current time using a given strategy.

    Parameters:
    -----------
    strategy_func : function
        Your strategy payoff function. Must return (S_range, payoff_at_expiration)
    strikes : list or tuple
        Initial strike prices
    K_names : list of str
        Names for each strike to display in textboxes
    current_price : float
        Current underlying price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    T_total : float
        Total time to expiration (years)
    """

    # Generate initial payoff at expiration
    S_range, payoff = strategy_func(current_price, *strikes, r, sigma, T_total)
    
    # Generate initial portfolio value (at t=0)
    def portfolio_value(t):
        T_remaining = max(T_total - t, 0)
        S_range, value = strategy_func(current_price, *strikes, r, sigma, T_remaining)
        return S_range, value

    S_port, value = portfolio_value(t=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.35)
    
    # Plot payoff and portfolio value
    line_payoff, = ax.plot(S_range, payoff, label="Payoff at Expiration", color="blue", lw=2)
    line_port, = ax.plot(S_port, value, label="Portfolio Value Now", color="orange", lw=2)
    ax.axvline(current_price, linestyle='--', color='gray', label=f'Current Price = {current_price}')
    ax.axhline(0, linewidth=1, color='black')
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Profit / Loss")
    ax.set_title(f"{strategy_func.__name__} Payoff and Portfolio Value")
    ax.grid(True)
    ax.legend()
    
    # Textboxes for strikes
    text_boxes = []
    for i, name in enumerate(K_names):
        axbox = plt.axes([0.15, 0.25 - i*0.05, 0.2, 0.03], facecolor="lightgoldenrodyellow")
        tb = TextBox(axbox, name, initial=str(strikes[i]))
        text_boxes.append(tb)

    # Slider for time t
    ax_slider = plt.axes([0.6, 0.15, 0.3, 0.03], facecolor='lightgray')
    slider = Slider(ax_slider, 'Time Elapsed', 0, T_total, valinit=0)

    # Update function for strikes or time
    def update(_):
        try:
            # Get updated strikes
            new_strikes = tuple(float(tb.text) for tb in text_boxes)
            t = slider.val
            T_remaining = max(T_total - t, 0)

            # Recompute payoff and portfolio value
            S_range, payoff = strategy_func(current_price, *new_strikes, r, sigma, T_total)
            S_port, value = strategy_func(current_price, *new_strikes, r, sigma, T_remaining)

            # Update lines
            line_payoff.set_xdata(S_range)
            line_payoff.set_ydata(payoff)
            line_port.set_xdata(S_port)
            line_port.set_ydata(value)

            # Rescale axes
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        except ValueError:
            print("Please enter valid numbers in the text boxes.")

    # Connect updates
    for tb in text_boxes:
        tb.on_submit(update)
    slider.on_changed(update)

    plt.show()

#%%


def portfolio_value(strategy, strikes, K_names, current_price=None, r=0.05, sigma=0.2, T=1.0)
# Example inputs
current_price = 100
strikes = [90, 110]
K_names = ["K1", "K2"]
r = 0.05
sigma = 0.2
T_total = 1.0  # 1 year to expiration

# Call the plotting function
plot_strategy_and_portfolio(
    strategy_func=bull_call_spread,
    strikes=strikes,
    K_names=K_names,
    current_price=current_price,
    r=r,
    sigma=sigma,
    T_total=T_total
)
