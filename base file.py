
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad



def black_scholes_delta(S, K, r, sigma, T, t):
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.000001))
    delta = norm.cdf(d1)
    return delta

def black_scholes_gamma(S, K, r, sigma, T, t):
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t)) 
        d2 = d1 - sigma * np.sqrt(T - t)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T - t))
        
    else: 
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.000001)) 
        d2 = d1 - sigma * np.sqrt(0.01)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(0.000001))
    return gamma

def black_scholes_vega(S, K, r, sigma, T, t):
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        vega = S * norm.pdf(d1) * np.sqrt(T - t)
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.000001))
        vega = S * norm.pdf(d1) * np.sqrt(0.000001)
    
    return vega

def black_scholes_zomma(S, K, r, sigma, T, t):
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
    else: 
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.000001))
        d2 = d1 - sigma * np.sqrt(0.000001)
    zomma = black_scholes_gamma(S, K, r, sigma, T, t) * (d1 * d2 - 1) / sigma
    return zomma

def black_scholes_vanna(S, K, r, sigma, T, t):
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
    else: 
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.000001))
        d2 = d1 - sigma * np.sqrt(0.000001)
    vanna = -norm.pdf(d1) * d2 / sigma
    return vanna

def black_scholes_volga(S, K, r, sigma, T, t):
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
        volga = S * norm.pdf(d1) * np.sqrt(T - t) * d1 * d2 / sigma
    else: 
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.0000001))
        d2 = d1 - sigma * np.sqrt(0.000001)
        volga = S * norm.pdf(d1) * np.sqrt(0.000001) * d1 * d2 / sigma
    return volga

def black_scholes_theta(S, K, r, sigma, T, t):
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
        theta_p = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T - t)) - r * K * np.exp(-r * (T - t)) * norm.cdf(d2)
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.000001))
        d2 = d1 - sigma * np.sqrt(0.0000001)
        theta_p = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(0.000001)) - r * K * np.exp(-r * (T - t)) * norm.cdf(d2)
    
    return theta_p

def black_scholes_charm(S, K, r, sigma, T, t):
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
        charm = -black_scholes_vega(S, K, r, sigma, T, t) * ((r * (T - t) - 0.5 * d2 * sigma * np.sqrt(T - t)) / (T - t) + d2 * sigma * np.sqrt(T - t) / (2 * (T - t)) - 1)
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.0000001))
        d2 = d1 - sigma * np.sqrt(0.0000001)
        charm = -black_scholes_vega(S, K, r, sigma, T, t) * ((r * (T - t) - 0.5 * d2 * sigma * np.sqrt(0.0000001)) / (0.000001) + d2 * sigma * np.sqrt(0.000001) / (2 * (0.000001)) - 1)
 
    return charm

def black_scholes_speed(S, K, r, sigma, T, t):
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(0.01)
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.0000001))
        d2 = d1 - sigma * np.sqrt(0.0000001)
   
    speed = -black_scholes_gamma(S, K, r, sigma, T, t) / S * (d1 / sigma)
    return speed

def black_scholes_veta(S, K, r, sigma, T, t):
    vega = black_scholes_vega(S, K, r, sigma, T, t)
    veta = -vega * t / (2 * (T - t)) if T-t !=0 else -vega * t / (2 * (0.0000001))
    return veta

def black_scholes_ultima(S, K, r, sigma, T, t):
    vega = black_scholes_vega(S, K, r, sigma, T, t)
    if T - t != 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(0.0000001))
        d2 = d1 - sigma * np.sqrt(0.0000001)
    
    ultima = -vega * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2) / sigma**2
    return ultima

def black_scholes_vegavanna(S, K, r, sigma, T, t):
    vega = black_scholes_vega(S, K, r, sigma, T, t)
    vanna = black_scholes_vanna(S, K, r, sigma, T, t)
    vegavanna = vega * vanna
    return vegavanna

# Function to calculate bull spread delta
def bull_call_spread_delta(S, K_long, K_short, r, sigma, T, t):
    delta_long = black_scholes_delta(S, K_long, r, sigma, T, t)
    delta_short = black_scholes_delta(S, K_short, r, sigma, T, t)
    delta = delta_long - delta_short
    return delta

# Function to calculate bull spread price
def bull_call_spread_price(S, K_long, K_short, r, sigma, T, t):
    price_long = black_scholes_price(S, K_long, r, sigma, T, t)
    price_short = black_scholes_price(S, K_short, r, sigma, T, t)
    price = price_long - price_short
    return price



# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


np.random.seed(42)

# Parameters
psi = 0.4
phi = 0.4
zeta = 0.5
sigma = 0.25
S0 = 100  # Initial stock price
T = 1  # Time to maturity (in years)
N = 365 * 24  # Number of hourly intervals
dt = 1 / N  # Time step
t = np.linspace(0, T, N+1, endpoint=False)  # Time array
quarter_intervals = int(N / 4)  # Number of intervals per quarter

# Function to calculate delta of call option using Black-Scholes model


def black_scholes_price(S, K, r, sigma, T, t):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t)) if T-t != 0 else (np.log(S / K) + (r + 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(0.0000001))
    d2 = d1 - sigma * np.sqrt(T - t) if T-t != 0 else  d1 - sigma * np.sqrt(0.0000001)
    price = S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    return price

# Function to calculate bull spread delta b
def bull_call_spread_delta(S, K_long, K_short, r, sigma, T, t):
    delta_long = black_scholes_delta(S, K_long, r, sigma, T, t)
    delta_short = black_scholes_delta(S, K_short, r, sigma, T, t)
    delta = delta_long - delta_short
    return delta

# Function to calculate bull spread price
def bull_call_spread_price(S, K_long, K_short, r, sigma, T, t):
    price_long = black_scholes_price(S, K_long, r, sigma, T, t)
    price_short = black_scholes_price(S, K_short, r, sigma, T, t)
    price = price_long - price_short
    return price



# Function to calculate M_delta
def calculate_delta_change(delta_func, i, dt):
    delta_i = delta_func(i * dt)
    delta_im1 = delta_func((i - 1) * dt)
    delta_change = (delta_i - delta_im1) / dt
    return delta_change

# Function to calculate M_delta
def calculate_M_delta(delta_func, i, il, phi, dt):
    integrand = lambda s: np.exp(-phi * (t[i-1] - s)) * calculate_delta_change(delta_func, s / dt, dt)
    M_delta, _ = quad(integrand, t[i - il], t[i-1])
    return M_delta


# Function to simulate stock price path with Mt
def simulate_stock_price_with_Mt(psi):

    
    S = np.zeros(N+1)
    delta = np.zeros(N+1)
    theta = np.zeros(N+1)
    S_strike_long = np.zeros(N+1)
    S_strike_short = np.zeros(N+1)
    spread_price = np.zeros(N+1)
    port_val = np.zeros(N+1)
    cash_ac = np.zeros(N+1)
    temp_delta = np.zeros(N+1)
    money_available = np.zeros(N+1)
    money_available[0] = 0
    money_invested = np.zeros(N+1)
    money_invested[0] = -100
    std_dev_previous = np.zeros(N+1)
    std_dev_previous[0] = sigma*np.sqrt(0.25)
    M_delta = np.zeros(N+1)
    temp = np.zeros(N+1)
    theta_is = np.zeros(N+1)
    

    S[0] = S0
    S_strike_long[0] = S[0]*(1-(sigma*np.sqrt(0.25)))
    S_strike_short[0] = S[0]*(1+(sigma*np.sqrt(0.25)))
    delta[0] = 1
    port_val[0] = 100
    temp_delta[0] = bull_call_spread_delta(S0, S0 * (1 - (sigma * np.sqrt(0.25))), S0 * (1 + (sigma * np.sqrt(0.25))), 0, sigma, 1/4, 0)  # Initial theta
    theta[0] = 1 / bull_call_spread_delta(S0, S0 * (1 - (sigma * np.sqrt(0.25))), S0 * (1 + (sigma * np.sqrt(0.25))), 0, sigma, 1/4, 0)
    cash_ac[0] = 0
    money_invested[0] = -100
    bull_call_spread_values = np.zeros(N+1)
    bull_call_spread_values[0] = bull_call_spread_price(S0, S0 * (1 - sigma * np.sqrt(0.25)), S0 * (1 + sigma * np.sqrt(0.25)), 0, sigma, 1/4, 0)
    M_delta[0] = 0
    theta_is[0] = 0


    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
        tolerance = 1e-10000  # Adjust this tolerance as needed
        
        spread_price = bull_call_spread_price(S[i], S_strike_long[i], S_strike_short[i], 0, sigma, 1/4, t[i % quarter_intervals])
        
        if i % quarter_intervals == 0  or i == 0:  # Check if it's the end of a quarter
                S_slice = S[i-quarter_intervals+1:i]
                std_dev = np.std(S_slice)
                std_dev_previous = std_dev
                S_strike_long[i] = S[i-1] - std_dev_previous # Ensure strike price is positive
                S_strike_short[i] = S[i-1] + std_dev_previous # Ensure strike price is positive
                
                # Calculate the amount of money available for buying new bull spreads
                money_available[i] = 0 if i == 0 else theta[i - quarter_intervals] * ((S[i-1] - S_strike_long[i - quarter_intervals]) if S[i] - S_strike_long[i - quarter_intervals] > 0 else 0) - ((S[i-1] - S_strike_short[i - quarter_intervals]) if S[i-1] - S_strike_short[i - quarter_intervals ] > 0 else 0)

                
                if money_available[i] <= (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001))/ bull_call_spread_delta(S[i-1], S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.001): 
                    theta[i] = 1 / bull_call_spread_delta(S[i-1], S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.00001)

                    cash_ac[i] = cash_ac[i-quarter_intervals] - theta[i] * (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)) + money_available[i]
                    
                else: 
                    theta[i] = money_available[i] / bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)
                    cash_ac[i] = cash_ac[i-quarter_intervals]
                    
                delta[i] = theta[i] * temp_delta[i]
                
                temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i-1], S_strike_short[i-1], 0, sigma, 1/4, 0.001)
                port_val[i] = cash_ac[i] + S[i-1] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.0001)

                M_delta[i] = calculate_M_delta(lambda s: -delta[int(s / dt)], i, quarter_intervals, phi, dt)
                
            
        else: 
                # Maintain values as set at the beginning of the quarter
                M_delta[i] = calculate_M_delta(lambda s: -delta[int(s / dt)], i, i % quarter_intervals, phi, dt)
                
                temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i-1], S_strike_short[i-1], 0, sigma, 1/4, (i % quarter_intervals)*dt)
                std_dev_previous = S[i - (i % quarter_intervals)]*(0.25*np.sqrt(0.25)) 
                S_strike_long[i] = S_strike_long[i - (i % quarter_intervals)] 
                S_strike_short[i] = S_strike_short[i - (i % quarter_intervals)] 
                money_available[i] = 0
                cash_ac[i] = cash_ac[i-(i % quarter_intervals)]
                theta[i] = theta[i-(i % quarter_intervals)]
                remaining_time = 1 - (i % quarter_intervals) * dt  # Remaining time until end of the quarter
                delta[i] = theta[i] * temp_delta[i]
                port_val[i] = cash_ac[i] + S[i-1] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, dt*(i % quarter_intervals))

            
        dS = psi * S[i-1] * M_delta[i-1] * dt  + sigma * S[i-1]**(1 - zeta) * dW  # Change in stock price with Mt
        S[i] = max(S[i-1] + dS, 0.01)  # Ensure stock price is positive
    # Calculate Greeks
    deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna = calculate_greeks_portfolio(S, theta, S_strike_long, S_strike_short, 0, sigma, T, t, psi, quarter_intervals=quarter_intervals)
    
    return S, delta, S_strike_long, S_strike_short, theta, cash_ac, deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna, port_val
     


# Function to calculate greeks for the entire year for the portfolio of the bull call spread
def calculate_greeks_portfolio(S, theta, S_strike_long, S_strike_short, r, sigma, T, t, psi, quarter_intervals):
    deltas = np.zeros(N+1)
    gamma = np.zeros(N+1)
    vega = np.zeros(N+1)
    zomma = np.zeros(N+1)
    vanna = np.zeros(N+1)
    volga = np.zeros(N+1)
    theta_p = np.zeros(N+1)
    charm = np.zeros(N+1)
    speed = np.zeros(N+1)
    veta = np.zeros(N+1)
    ultima = np.zeros(N+1)
    vegavanna = np.zeros(N+1)

    for i in range(1, N+1):
        time_to_maturity = 1/4 - t[(i % quarter_intervals)]  # Adjusted time to maturity for quarterly intervals
        deltas[i] = theta[i] * (bull_call_spread_delta(S[i], S_strike_long[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        gamma[i] = theta[i] * (black_scholes_gamma(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_gamma(S[i% quarter_intervals], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        vega[i] = theta[i] * (black_scholes_vega(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_vega(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        zomma[i] = theta[i] * (black_scholes_zomma(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_zomma(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        vanna[i] = theta[i] * (black_scholes_vanna(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_vanna(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        volga[i] = theta[i] * (black_scholes_volga(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_volga(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        theta_p[i] = theta[i] * (black_scholes_theta(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_theta(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        charm[i] = theta[i] * (black_scholes_charm(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_charm(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        speed[i] = theta[i] * (black_scholes_speed(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_speed(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        veta[i] = theta[i] * (black_scholes_veta(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_veta(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        ultima[i] = theta[i] * (black_scholes_ultima(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_ultima(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))
        vegavanna[i] = theta[i] * (black_scholes_vegavanna(S[i], S_strike_long[i], r, sigma, T/4, t[i% quarter_intervals]) - black_scholes_vegavanna(S[i], S_strike_short[i], r, sigma, T/4, t[i% quarter_intervals]))

    return deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna



def simulate_stock_price_without_Mt():
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
        dS = sigma * S[i-1]**(1 - zeta) * dW  # Change in stock price without Mt
        S[i] = S[i-1] + dS  # Ensure stock price is positive
    return S





def calculate_greeks(S, theta, K_long, K_short, r, sigma, T, t, quarter_intervals):
    deltas = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    gamma = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    vega = np.zeros(quarter_intervals+1)   # Adjusted to calculate only for the first quarter
    zomma = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    vanna = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    volga = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    theta_p = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    charm = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    speed = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    veta = np.zeros(quarter_intervals+1)   # Adjusted to calculate only for the first quarter
    ultima = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    vegavanna = np.zeros(quarter_intervals+1)  # Adjusted to calculate only for the first quarter
    
    for i in range(quarter_intervals+1):  # Adjusted to iterate only over the first quarter
        deltas[i] = theta[i] * (black_scholes_delta(S[i], K_long, r, sigma, T, t[i]) - black_scholes_delta(S[i], K_short, r, sigma, T, t[i]))
        gamma[i] =theta[i] * ( black_scholes_gamma(S[i], K_long, r, sigma, T, t[i]) - black_scholes_gamma(S[i], K_short, r, sigma, T, t[i]))
        vega[i] = theta[i] * (black_scholes_vega(S[i], K_long, r, sigma, T, t[i]) - black_scholes_vega(S[i], K_short, r, sigma, T, t[i]))
        zomma[i] = theta[i] * (black_scholes_zomma(S[i], K_long, r, sigma, T, t[i]) - black_scholes_zomma(S[i], K_short, r, sigma, T, t[i]))
        vanna[i] = theta[i] * (black_scholes_vanna(S[i], K_long, r, sigma, T, t[i]) - black_scholes_vanna(S[i], K_short, r, sigma, T, t[i]))
        volga[i] = theta[i] * (black_scholes_volga(S[i], K_long, r, sigma, T, t[i]) - black_scholes_volga(S[i], K_short, r, sigma, T, t[i]))
        theta_p[i] = theta[i] * (black_scholes_theta(S[i], K_long, r, sigma, T, t[i]) - black_scholes_theta(S[i], K_short, r, sigma, T, t[i]))
        charm[i] = theta[i] * (black_scholes_charm(S[i], K_long, r, sigma, T, t[i]) - black_scholes_charm(S[i], K_short, r, sigma, T, t[i]))
        speed[i] = theta[i] * (black_scholes_speed(S[i], K_long, r, sigma, T, t[i]) - black_scholes_speed(S[i], K_short, r, sigma, T, t[i]))
        veta[i] = theta[i] * (black_scholes_veta(S[i], K_long, r, sigma, T, t[i]) - black_scholes_veta(S[i], K_short, r, sigma, T, t[i]))
        ultima[i] = theta[i] * (black_scholes_ultima(S[i], K_long, r, sigma, T, t[i]) - black_scholes_ultima(S[i], K_short, r, sigma, T, t[i]))
        vegavanna[i] = theta[i] * (black_scholes_vegavanna(S[i], K_long, r, sigma, T, t[i]) - black_scholes_vegavanna(S[i], K_short, r, sigma, T, t[i]))
    
    return deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna

psi_values = [0.2, 0.4, 0.6, 0.8]

for psi in psi_values:
    S, delta, S_strike_long, S_strike_short, theta, cash_ac, deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna, _ = simulate_stock_price_with_Mt(psi)
   



#Plots; 


for psi in psi_values:
    S, delta, S_strike_long, S_strike_short, theta, cash_ac, deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna, _ = simulate_stock_price_with_Mt(psi)
   

    # Plotting greeks
    fig, axs = plt.subplots(5, 3, figsize=(18, 12))

    axs[0, 0].plot(t, deltas)
    axs[0, 0].set_title('spread Delta')
    axs[0, 1].plot(t, gamma)
    axs[0, 1].set_title('Gamma')
    axs[0, 2].plot(t, vega)
    axs[0, 2].set_title('Vega')

    axs[1, 0].plot(t, zomma)
    axs[1, 0].set_title('Zomma')
    axs[1, 1].plot(t, vanna)
    axs[1, 1].set_title('Vanna')
    axs[1, 2].plot(t, volga)
    axs[1, 2].set_title('Volga')

    axs[2, 0].plot(t, theta_p)
    axs[2, 0].set_title('Theta')
    axs[2, 1].plot(t, charm)
    axs[2, 1].set_title('Charm')
    axs[2, 2].plot(t, speed)
    axs[2, 2].set_title('Speed')

    axs[3, 0].plot(t, veta)
    axs[3, 0].set_title('Veta')
    axs[3, 1].plot(t, ultima)
    axs[3, 1].set_title('Ultima')
    axs[3, 2].plot(t, vegavanna)
    axs[3, 2].set_title('Vegavanna')

    axs[4, 0].plot(t, theta)
    axs[4, 0].set_title('Theta - number of portfolio of options')
    axs[4, 1].plot(t, S)
    axs[4, 1].set_title('stock price')
    axs[4, 2].plot(t, cash_ac)
    axs[4, 2].set_title('cash_a/c')

    fig.suptitle(f'Psi = {psi}')  # Title indicating the value of psi
    plt.tight_layout()
    plt.show()

# Plot stock price paths with and without Mt for all psi values
plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=1):
    plt.subplot(3, len(psi_values), i)
    stock_price_with_Mt, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _= simulate_stock_price_with_Mt(psi)
    stock_price_without_Mt = simulate_stock_price_without_Mt()
    plt.plot(t, stock_price_with_Mt, label='With Mt', color='blue')
    plt.plot(t, stock_price_without_Mt, label='Without Mt', linestyle='--', color='red')
    plt.xlabel('Time (Years)')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Paths (Psi = {psi})')
    plt.legend()

# Plot delta values for all psi values
plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=len(psi_values)+1):
    plt.subplot(3, len(psi_values), i)
    S, delta, S_strike_long, S_strike_short, theta, cash_ac, _,  _, _, _, _, _, _, _, _, _, _, _, _ = simulate_stock_price_with_Mt(psi)
    deltas, _, _, _, _, _, _, _, _, _, _, _= calculate_greeks(S, theta, S0 - sigma*np.sqrt(0.25), S0 + sigma*np.sqrt(0.25), 0, sigma, T/4, t, quarter_intervals = quarter_intervals)
    plt.plot(t[:quarter_intervals+1], deltas)
    plt.xlabel('Time (Years)')
    plt.ylabel('Delta')
    plt.title(f'Delta (Psi = {psi})')

# Plot cash account values with and without Mt for all psi values
plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=2*len(psi_values)+1):
    plt.subplot(3, len(psi_values), i)
    _, _, _, _, _, _,_, _, _, _, _, _, _, _, _, _, _, _, port_val = simulate_stock_price_with_Mt(psi)
    plt.plot(t, cash_ac, label='With Mt', color='blue')
    plt.plot(t, cash_ac, label='Without Mt', linestyle='--', color='red')
    plt.xlabel('Time (Years)')
    plt.ylabel('MSF Portfolio value')
    plt.title(f'Cash Account Value (Psi = {psi})')
    plt.legend()
    


# rolling fund's portfolio value 

# Plot cash account values with and without Mt for all psi values
plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=2*len(psi_values)+1):
    plt.subplot(3, len(psi_values), i)
    _, _, _, _, _, _,_, _, _, _, _, _, _, _, _, _, _, _, port_val = simulate_stock_price_with_Mt(psi)
    plt.plot(t, port_val, label='With Mt', color='blue')
    plt.plot(t, port_val, label='Without Mt', linestyle='--', color='red')
    plt.xlabel('Time (Years)')
    plt.ylabel('MSF Portfolio value')
    plt.title(f'MSF Rolling Portfolio Value (Psi = {psi})')
    plt.legend()


# MC 
# Parameters
num_paths = 50

# Generate Monte Carlo paths using the provided function
paths = np.zeros((num_paths, N+1))
for i in range(num_paths):
        S, delta, S_strike_long, S_strike_short, theta, cash_ac, deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna, port_val
        paths[i],_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _= simulate_stock_price_with_Mt()


# Calculate quadratic variation
quadratic_variation = np.sum(np.diff(paths, axis=1)**2, axis=1)

# Assign colors based on quadratic variation
colors = np.where(quadratic_variation > np.median(quadratic_variation), 'red', 'blue')

# Plot paths
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

for i in range(num_paths):
    ax.plot(range(N + 1), paths[i], zs=range(N + 1), color=colors[i])

ax.set_xlabel('Time Steps')
ax.set_ylabel('Stock Price')
ax.set_zlabel('Time')

plt.title('Monte Carlo Paths with Different Quadratic Variation')
plt.show()


# Plot paths
plt.figure(figsize=(10, 6))
for i in range(num_paths):
    plt.plot(range(N + 1), paths[i], color=colors[i])

plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Monte Carlo P aths with Different Quadratic Variation')
plt.show()

 # general plot

# Plot paths
plt.figure(figsize=(10, 6))
for i in range(num_paths):
    plt.plot(range(N + 1), paths[i])

plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Monte Carlo Paths')
plt.show()


# conditional MC stats for prob in ending up in a region

#D Conditional stats

import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_summary(N, num_paths=50):
    # Generate Monte Carlo paths using the provided function
    paths = np.zeros((num_paths, N+1))
    for i in range(num_paths):
        paths[i] = simulate_stock_price_with_Mt(0.1)

    # Filter paths where the final stock price is greater than 500
    filtered_paths = paths[paths[:, -1] > 500]

    # Calculate quadratic variation for filtered paths
    quadratic_variation_filtered = np.sum(np.diff(filtered_paths, axis=1)**2, axis=1)

    # Calculate statistics for filtered paths
    mean_filtered = np.mean(filtered_paths, axis=0)
    std_deviation_filtered = np.std(filtered_paths, axis=0)
    minimum_filtered = np.min(filtered_paths, axis=0)
    maximum_filtered = np.max(filtered_paths, axis=0)
    percentiles_filtered = np.percentile(filtered_paths, [25, 50, 75], axis=0)
    outliers_filtered = filtered_paths[(filtered_paths < np.percentile(filtered_paths, 25, axis=0)) | 
                                       (filtered_paths > np.percentile(filtered_paths, 75, axis=0))]

    # Calculate probability of ST ending up higher than 500
    probability_ST_gt_500 = len(filtered_paths) / num_paths

    # Plot paths
    plt.figure(figsize=(10, 6))
    for i in range(num_paths):
        plt.plot(range(N + 1), paths[i], color='blue', alpha=0.3)

    for i in range(len(filtered_paths)):
        plt.plot(range(N + 1), filtered_paths[i], color='red')

    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title('Monte Carlo Paths with Final Stock Price > 500 (N={})'.format(N))
    plt.show()

    # Print statistics
    print("Statistics for paths where final stock price > 500 (N={}):".format(N))
    print("Mean:", mean_filtered)
    print("Standard Deviation:", std_deviation_filtered)
    print("Minimum:", minimum_filtered)
    print("Maximum:", maximum_filtered)
    print("25th, 50th, and 75th Percentiles:", percentiles_filtered)
    print("Outliers:", outliers_filtered)
    print("Probability of ST ending up higher than 500:", probability_ST_gt_500)
    print("\n")

# Define different values of N
Ns = [365*24, 365, 48, 12, 3]

# Generate summaries for each value of N
for N in Ns:
    monte_carlo_summary(N)



# Simulate S with LVOL STICKY DELTA - trial one (theta not changes per skew)
def simulate_stock_price_with_Mt_LVOL_thetaunchanged(psi):

    
    S = np.zeros(N+1)
    delta = np.zeros(N+1)
    theta = np.zeros(N+1)
    S_strike_long = np.zeros(N+1)
    S_strike_short = np.zeros(N+1)
    spread_price = np.zeros(N+1)
    port_val = np.zeros(N+1)
    cash_ac = np.zeros(N+1)
    temp_delta = np.zeros(N+1)
    money_available = np.zeros(N+1)
    money_available[0] = 0
    money_invested = np.zeros(N+1)
    money_invested[0] = -100
    std_dev_previous = np.zeros(N+1)
    std_dev_previous[0] = sigma*np.sqrt(0.25)
    M_delta = np.zeros(N+1)
    temp = np.zeros(N+1)
    theta_is = np.zeros(N+1)
    

    S[0] = S0
    S_strike_long[0] = S[0]*(1-(sigma*np.sqrt(0.25)))
    S_strike_short[0] = S[0]*(1+(sigma*np.sqrt(0.25)))
    delta[0] = 1
    port_val[0] = 100
    temp_delta[0] = bull_call_spread_delta(S0, S0 * (1 - (sigma * np.sqrt(0.25))), S0 * (1 + (sigma * np.sqrt(0.25))), 0, sigma, 1/4, 0)  # Initial theta
    theta[0] = 1 / bull_call_spread_delta(S0, S0 * (1 - (sigma * np.sqrt(0.25))), S0 * (1 + (sigma * np.sqrt(0.25))), 0, sigma, 1/4, 0)
    cash_ac[0] = 0
    money_invested[0] = -100
    bull_call_spread_values = np.zeros(N+1)
    bull_call_spread_values[0] = bull_call_spread_price(S0, S0 * (1 - sigma * np.sqrt(0.25)), S0 * (1 + sigma * np.sqrt(0.25)), 0, sigma, 1/4, 0)
    M_delta[0] = 0
    theta_is[0] = 0


    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
        tolerance = 1e-10000  # Adjust this tolerance as needed
        
        spread_price = bull_call_spread_price(S[i], S_strike_long[i], S_strike_short[i], 0, sigma, 1/4, t[i % quarter_intervals])
        
        if i % quarter_intervals == 0  or i == 0:  # Check if it's the end of a quarter
                S_slice = S[i-quarter_intervals+1:i]
                std_dev = np.std(S_slice)
                std_dev_previous = std_dev
                S_strike_long[i] = S[i-1] - std_dev_previous # Ensure strike price is positive
                S_strike_short[i] = S[i-1] + std_dev_previous # Ensure strike price is positive
                
                # Calculate the amount of money available for buying new bull spreads
                money_available[i] = 0 if i == 0 else theta[i - quarter_intervals] * ((S[i-1] - S_strike_long[i - quarter_intervals]) if S[i] - S_strike_long[i - quarter_intervals] > 0 else 0) - ((S[i-1] - S_strike_short[i - quarter_intervals]) if S[i-1] - S_strike_short[i - quarter_intervals ] > 0 else 0)

                
                if money_available[i] <= (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001))/ bull_call_spread_delta(S[i-1], S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.001): 
                    theta[i] = 1 / bull_call_spread_delta(S[i-1], S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.00001)

                    cash_ac[i] = cash_ac[i-quarter_intervals] - theta[i] * (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)) + money_available[i]
                    
                else: 
                    theta[i] = money_available[i] / bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)
                    cash_ac[i] = cash_ac[i-quarter_intervals]
                    
                delta[i] = theta[i] * temp_delta[i]
                
                temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i-1], S_strike_short[i-1], 0, sigma, 1/4, 0.00001) + ((black_scholes_vega(S[i-1], S_strike_long[i], 0, sigma, 1/4, 0.00001)-black_scholes_vega(S[i-1], S_strike_short[i], 0, sigma, 1/4, 0.00001))*(-zeta*beta*np.exp(T-(dt*(i%quarter_intervals))/((S[i-1]+S_strike_long[i])**(zeta+1)))))
                port_val[i] = cash_ac[i] + S[i-1] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.0001)

                M_delta[i] = calculate_M_delta(lambda s: -delta[int(s / dt)], i, quarter_intervals, phi, dt)
                 
            
        else: 
                # Maintain values as set at the beginning of the quarter
                M_delta[i] = calculate_M_delta(lambda s: -delta[int(s / dt)], i, i % quarter_intervals, phi, dt)
                
                temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i], S_strike_short[i], 0, sigma, 1/4,t[i % quarter_intervals]) + ((black_scholes_vega(S[i-1], S_strike_long[i], 0, sigma, 1/4, 0.00001)-black_scholes_vega(S[i-1], S_strike_short[i], 0, sigma, 1/4, 0.00001))*(-zeta*beta*np.exp(T-(dt*(i%quarter_intervals))/((S[i-1]+S_strike_long[i])**(zeta+1)))))
                std_dev_previous = S[i - (i % quarter_intervals)]*(0.25*np.sqrt(0.25)) 
                S_strike_long[i] = S_strike_long[i - (i % quarter_intervals)] 
                S_strike_short[i] = S_strike_short[i - (i % quarter_intervals)] 
                money_available[i] = 0
                cash_ac[i] = cash_ac[i-(i % quarter_intervals)]
                theta[i] = theta[i-(i % quarter_intervals)]
                remaining_time = 1 - (i % quarter_intervals) * dt  # Remaining time until end of the quarter
                delta[i] = theta[i] * temp_delta[i]
                port_val[i] = cash_ac[i] + S[i-1] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, dt*(i % quarter_intervals))

            
        dS = psi * S[i-1] * M_delta[i-1] * dt  + sigma * S[i-1]**(1 - zeta) * dW  # Change in stock price with Mt
        S[i] = max(S[i-1] + dS, 0.01)  # Ensure stock price is positive
    # Calculate Greeks
    deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna = calculate_greeks_portfolio(S, theta, S_strike_long, S_strike_short, 0, sigma, T, t, psi, quarter_intervals=quarter_intervals)
    
    return S, delta, S_strike_long, S_strike_short, theta, cash_ac, deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna, port_val



# MSF value with and without Mt (BSM) and (LVOL)




# Simulate S with LVOL STICKY DELTA 
def simulate_stock_price_with_Mt_LVOL(psi):

    
    S = np.zeros(N+1)
    delta = np.zeros(N+1)
    theta = np.zeros(N+1)
    S_strike_long = np.zeros(N+1)
    S_strike_short = np.zeros(N+1)
    spread_price = np.zeros(N+1)
    port_val = np.zeros(N+1)
    cash_ac = np.zeros(N+1)
    temp_delta = np.zeros(N+1)
    money_available = np.zeros(N+1)
    money_available[0] = 0
    money_invested = np.zeros(N+1)
    money_invested[0] = -100
    std_dev_previous = np.zeros(N+1)
    std_dev_previous[0] = sigma*np.sqrt(0.25)
    M_delta = np.zeros(N+1)
    temp = np.zeros(N+1)
    theta_is = np.zeros(N+1)
    

    S[0] = S0
    S_strike_long[0] = S[0]*(1-(sigma*np.sqrt(0.25)))
    S_strike_short[0] = S[0]*(1+(sigma*np.sqrt(0.25)))
    delta[0] = 1
    port_val[0] = 100
    temp_delta[0] = bull_call_spread_delta(S0, S0 * (1 - (sigma * np.sqrt(0.25))), S0 * (1 + (sigma * np.sqrt(0.25))), 0, sigma, 1/4, 0.0000001) +  (black_scholes_vega(S0, S_strike_long[0], 0, sigma, 1/4, 0.00001)-black_scholes_vega(S0, S_strike_long[0], 0, sigma, 1/4, 0.00001))*(-zeta*beta*np.exp(T)/((S0 +  S0 * (1 - (sigma * np.sqrt(0.25))))**(zeta+1)))  # Initial theta
    theta[0] = 1 / temp_delta[0]
    cash_ac[0] = 0
    money_invested[0] = -100
    bull_call_spread_values = np.zeros(N+1)
    bull_call_spread_values[0] = bull_call_spread_price(S0, S0 * (1 - sigma * np.sqrt(0.25)), S0 * (1 + sigma * np.sqrt(0.25)), 0, sigma, 1/4, 0)
    M_delta[0] = 0
    theta_is[0] = 0


    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
        tolerance = 1e-10000  # Adjust this tolerance as needed
        
        spread_price = bull_call_spread_price(S[i], S_strike_long[i], S_strike_short[i], 0, sigma, 1/4, t[i % quarter_intervals])
        
        if i % quarter_intervals == 0  or i == 0:  # Check if it's the end of a quarter
                S_slice = S[i-quarter_intervals+1:i]
                std_dev = np.std(S_slice)
                std_dev_previous = std_dev
                S_strike_long[i] = S[i-1] - std_dev_previous # Ensure strike price is positive
                S_strike_short[i] = S[i-1] + std_dev_previous # Ensure strike price is positive
                
                # Calculate the amount of money available for buying new bull spreads
                money_available[i] = 0 if i == 0 else theta[i - quarter_intervals] * ((S[i-1] - S_strike_long[i - quarter_intervals]) if S[i] - S_strike_long[i - quarter_intervals] > 0 else 0) - ((S[i-1] - S_strike_short[i - quarter_intervals]) if S[i-1] - S_strike_short[i - quarter_intervals ] > 0 else 0)

                
                if money_available[i] <= (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001))/temp_delta[i]: 
                    theta[i] = 1 / temp_delta[i]
                    cash_ac[i] = cash_ac[i-quarter_intervals] - theta[i] * (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)) + money_available[i]
                    
                else: 
                    theta[i] = money_available[i] / bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)
                    cash_ac[i] = cash_ac[i-quarter_intervals]
                    
                delta[i] = theta[i] * temp_delta[i]
                
                temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i-1], S_strike_short[i-1], 0, sigma, 1/4, 0.00001) + ((black_scholes_vega(S[i-1], S_strike_long[i], 0, sigma, 1/4, 0.00001)-black_scholes_vega(S[i-1], S_strike_short[i], 0, sigma, 1/4, 0.00001))*(-zeta*beta*np.exp(T-(dt*(i%quarter_intervals))/((S[i-1]+S_strike_long[i])**(zeta+1)))))
                port_val[i] = cash_ac[i] + S[i-1] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.0001)

                M_delta[i] = calculate_M_delta(lambda s: -delta[int(s / dt)], i, quarter_intervals, phi, dt)
                 
            
        else: 
                # Maintain values as set at the beginning of the quarter
                M_delta[i] = calculate_M_delta(lambda s: -delta[int(s / dt)], i, i % quarter_intervals, phi, dt)
                
                temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i], S_strike_short[i], 0, sigma, 1/4,t[i % quarter_intervals]) + ((black_scholes_vega(S[i-1], S_strike_long[i], 0, sigma, 1/4, 0.00001)-black_scholes_vega(S[i-1], S_strike_short[i], 0, sigma, 1/4, 0.00001))*(-zeta*beta*np.exp(T-(dt*(i%quarter_intervals))/((S[i-1]+S_strike_long[i])**(zeta+1)))))
                std_dev_previous = S[i - (i % quarter_intervals)]*(0.25*np.sqrt(0.25)) 
                S_strike_long[i] = S_strike_long[i - (i % quarter_intervals)] 
                S_strike_short[i] = S_strike_short[i - (i % quarter_intervals)] 
                money_available[i] = 0
                cash_ac[i] = cash_ac[i-(i % quarter_intervals)]
                theta[i] = theta[i-(i % quarter_intervals)]
                remaining_time = 1 - (i % quarter_intervals) * dt  # Remaining time until end of the quarter
                delta[i] = theta[i] * temp_delta[i]
                port_val[i] = cash_ac[i] + S[i-1] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, dt*(i % quarter_intervals))

            
        dS = psi * S[i-1] * M_delta[i-1] * dt  + sigma * S[i-1]**(1 - zeta) * dW  # Change in stock price with Mt
        S[i] = max(S[i-1] + dS, 0.01)  # Ensure stock price is positive
    # Calculate Greeks
    deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna = calculate_greeks_portfolio(S, theta, S_strike_long, S_strike_short, 0, sigma, T, t, psi, quarter_intervals=quarter_intervals)
    
    return S, delta, S_strike_long, S_strike_short, theta, cash_ac, deltas, gamma, vega, zomma, vanna, volga, theta_p, charm, speed, veta, ultima, vegavanna, port_val


def simulate_stock_price_without_Mt_MSF():
    S = np.zeros(N+1)
    delta = np.zeros(N+1)
    theta = np.zeros(N+1)
    S_strike_long = np.zeros(N+1)
    S_strike_short = np.zeros(N+1)
    spread_price = np.zeros(N+1)
    port_val = np.zeros(N+1)
    cash_ac = np.zeros(N+1)
    temp_delta = np.zeros(N+1)
    money_available = np.zeros(N+1)
    money_available[0] = 0
    money_invested = np.zeros(N+1)
    money_invested[0] = -100
    std_dev_previous = np.zeros(N+1)
    std_dev_previous[0] = sigma*np.sqrt(0.25)
    M_delta = np.zeros(N+1)
    temp = np.zeros(N+1)
    theta_is = np.zeros(N+1)
    

    S[0] = S0
    S_strike_long[0] = S[0]*(1-(sigma*np.sqrt(0.25)))
    S_strike_short[0] = S[0]*(1+(sigma*np.sqrt(0.25)))
    delta[0] = 1
    port_val[0] = 100
    temp_delta[0] = bull_call_spread_delta(S0, S0 * (1 - (sigma * np.sqrt(0.25))), S0 * (1 + (sigma * np.sqrt(0.25))), 0, sigma, 1/4, 0)  # Initial theta
    theta[0] = 1 / bull_call_spread_delta(S0, S0 * (1 - (sigma * np.sqrt(0.25))), S0 * (1 + (sigma * np.sqrt(0.25))), 0, sigma, 1/4, 0)
    cash_ac[0] = 0
    money_invested[0] = -100
    bull_call_spread_values = np.zeros(N+1)
    bull_call_spread_values[0] = bull_call_spread_price(S0, S0 * (1 - sigma * np.sqrt(0.25)), S0 * (1 + sigma * np.sqrt(0.25)), 0, sigma, 1/4, 0)
    M_delta[0] = 0
    theta_is[0] = 0
    
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
        tolerance = 1e-10000  # Adjust this tolerance as needed
        dS = sigma * S[i-1]**(1 - zeta) * dW  # Change in stock price without Mt
        S[i] = S[i-1] + dS  # Ensure stock price is positive

        spread_price = bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_short[i], 0, sigma, 1/4, max(t[i % quarter_intervals],0.0000001))
        if abs(i % quarter_intervals == 0) < tolerance or i == 0:  # Check if it's the end of a quarter
                S_slice = S[i-quarter_intervals+1:i-2]
                std_dev = np.std(S_slice)
                std_dev_previous = std_dev
                S_strike_long[i] = S[i-1] - std_dev_previous # Ensure strike price is positive
                S_strike_short[i] = S[i-1] + std_dev_previous # Ensure strike price is positive
                
                # Calculate the amount of money available for buying new bull spreads
                money_available[i] = 0 if i == 0 else theta[i - quarter_intervals] * ((S[i-1] - S_strike_long[i - quarter_intervals]) if S[i] - S_strike_long[i - quarter_intervals] > 0 else 0) - ((S[i-1] - S_strike_short[i - quarter_intervals]) if S[i-1] - S_strike_short[i - quarter_intervals ] > 0 else 0)

                
                if money_available[i] <= (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.000001))/ bull_call_spread_delta(S[i-1], S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.0000001): 
                    theta[i] = 1 / bull_call_spread_delta(S[i-1], S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.00001)

                    cash_ac[i] = cash_ac[i%quarter_intervals] - theta[i] * (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)) + money_available[i]
                    
                else: 
                    theta[i] = money_available[i] / bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)
                    cash_ac[i] = cash_ac[i%quarter_intervals]
                    
                delta[i] = theta[i] * temp_delta[i]
                
                temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i-1], S_strike_short[i-1], 0, sigma, 1/4, 0.0000001)
                port_val[i] = cash_ac[i] + S[i] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.0000001) if i == 0 else 100

            
        
        else: 
                # Maintain values as set at the beginning of the quarter
                
                temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i-1], S_strike_short[i-1], 0, sigma, 1/4, (i % quarter_intervals)*dt)
                std_dev_previous = S[i - (i % quarter_intervals)]*(0.25*np.sqrt(0.25)) 
                S_strike_long[i] = S_strike_long[i - (i % quarter_intervals)] 
                S_strike_short[i] = S_strike_short[i - (i % quarter_intervals)] 
                money_available[i] = 0
                cash_ac[i] = cash_ac[i-(i % quarter_intervals)]
                theta[i] = theta[i-(i % quarter_intervals)]
                remaining_time = 1 - (i % quarter_intervals) * dt  # Remaining time until end of the quarter
                delta[i] = theta[i] * temp_delta[i]
                port_val[i] = cash_ac[i] + S[i] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, dt*(i % quarter_intervals))

    return S, cash_ac, port_val, theta

def simulate_stock_price_without_Mt_MSF_LVOL():
    S = np.zeros(N+1)
    delta = np.zeros(N+1)
    theta = np.zeros(N+1)
    S_strike_long = np.zeros(N+1)
    S_strike_short = np.zeros(N+1)
    spread_price = np.zeros(N+1)
    port_val = np.zeros(N+1)
    cash_ac = np.zeros(N+1)
    temp_delta = np.zeros(N+1)
    money_available = np.zeros(N+1)
    money_available[0] = 0
    money_invested = np.zeros(N+1)
    money_invested[0] = -100
    std_dev_previous = np.zeros(N+1)
    std_dev_previous[0] = sigma*np.sqrt(0.25)
    M_delta = np.zeros(N+1)
    temp = np.zeros(N+1)
    theta_is = np.zeros(N+1)
    

    S[0] = S0
    S_strike_long[0] = S[0]*(1-(sigma*np.sqrt(0.25)))
    S_strike_short[0] = S[0]*(1+(sigma*np.sqrt(0.25)))
    delta[0] = 1
    port_val[0] = 100
    temp_delta[0] = bull_call_spread_delta(S0, S0 * (1 - (sigma * np.sqrt(0.25))), S0 * (1 + (sigma * np.sqrt(0.25))), 0, sigma, 1/4, 0.0000001) +  (black_scholes_vega(S0, S_strike_long[0], 0, sigma, 1/4, 0.00001)-black_scholes_vega(S0, S_strike_long[0], 0, sigma, 1/4, 0.00001))*(-zeta*beta*np.exp(T)/((S0 +  S0 * (1 - (sigma * np.sqrt(0.25))))**(zeta+1)))  # Initial theta
    theta[0] = 1 / temp_delta[0]
    cash_ac[0] = 0
    money_invested[0] = -100
    bull_call_spread_values = np.zeros(N+1)
    bull_call_spread_values[0] = bull_call_spread_price(S0, S0 * (1 - sigma * np.sqrt(0.25)), S0 * (1 + sigma * np.sqrt(0.25)), 0, sigma, 1/4, 0)
    M_delta[0] = 0
    theta_is[0] = 0
    
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
        tolerance = 1e-10000  # Adjust this tolerance as needed
        dS = sigma * S[i-1]**(1 - zeta) * dW  # Change in stock price without Mt
        S[i] = S[i-1] + dS  # Ensure stock price is positive

        spread_price = bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_short[i], 0, sigma, 1/4, max(t[i % quarter_intervals],0.0000001))
        if abs(i % quarter_intervals == 0) < tolerance or i == 0:  # Check if it's the end of a quarter
                S_slice = S[i-quarter_intervals+1:i-2]
                std_dev = np.std(S_slice)
                std_dev_previous = std_dev
                S_strike_long[i] = S[i-1] - std_dev_previous # Ensure strike price is positive
                S_strike_short[i] = S[i-1] + std_dev_previous # Ensure strike price is positive
                
                # Calculate the amount of money available for buying new bull spreads
                money_available[i] = 0 if i == 0 else theta[i - quarter_intervals] * ((S[i-1] - S_strike_long[i - quarter_intervals]) if S[i] - S_strike_long[i - quarter_intervals] > 0 else 0) - ((S[i-1] - S_strike_short[i - quarter_intervals]) if S[i-1] - S_strike_short[i - quarter_intervals ] > 0 else 0)

                
                if money_available[i] <= (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.000001))/ temp_delta[i]: 
                    theta[i] = 1 / temp_delta[i]

                    cash_ac[i] = cash_ac[i%quarter_intervals] - theta[i] * (bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)) + money_available[i]
                    
                else: 
                    theta[i] = money_available[i] / bull_call_spread_price(S[i-1], S_strike_long[i], S_strike_long[i], 0, sigma, 1/4, 0.001)
                    cash_ac[i] = cash_ac[i%quarter_intervals]
                    
                delta[i] = theta[i] * temp_delta[i]
                
                temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i-1], S_strike_short[i-1], 0, sigma, 1/4, 0.00001) + ((black_scholes_vega(S[i-1], S_strike_long[i], 0, sigma, 1/4, 0.00001)-black_scholes_vega(S[i-1], S_strike_short[i], 0, sigma, 1/4, 0.00001))*(-zeta*beta*np.exp(T-(dt*(i%quarter_intervals))/((S[i-1]+S_strike_long[i])**(zeta+1)))))
                port_val[i] = cash_ac[i] + S[i] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, 0.0000001) if i == 0 else 100

                
            
        else: 
            # Maintain values as set at the beginning of the quarter
                
            temp_delta[i] = bull_call_spread_delta(S[i-1], S_strike_long[i], S_strike_short[i], 0, sigma, 1/4,t[i % quarter_intervals]) + ((black_scholes_vega(S[i-1], S_strike_long[i], 0, sigma, 1/4, 0.00001)-black_scholes_vega(S[i-1], S_strike_short[i], 0, sigma, 1/4, 0.00001))*(-zeta*beta*np.exp(T-(dt*(i%quarter_intervals))/((S[i-1]+S_strike_long[i])**(zeta+1)))))
            std_dev_previous = S[i - (i % quarter_intervals)]*(0.25*np.sqrt(0.25)) 
            S_strike_long[i] = S_strike_long[i - (i % quarter_intervals)] 
            S_strike_short[i] = S_strike_short[i - (i % quarter_intervals)] 
            money_available[i] = 0
            cash_ac[i] = cash_ac[i-(i % quarter_intervals)]
            theta[i] = theta[i-(i % quarter_intervals)]
            remaining_time = 1 - (i % quarter_intervals) * dt  # Remaining time until end of the quarter
            delta[i] = theta[i] * temp_delta[i]
            port_val[i] = cash_ac[i] + S[i] + theta[i]* bull_call_spread_price(S[i-1],S_strike_long[i],S_strike_short[i], 0, sigma, 1/4, dt*(i % quarter_intervals))

    return S, cash_ac, port_val, theta




plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=2*len(psi_values)+1):
    plt.subplot(3, len(psi_values), i)
    _, _, _, _, _, _,_, _, _, _, _, _, _, _, _, _, _, _, port_val = simulate_stock_price_with_Mt(psi)
    _, _, port_val_without, _ = simulate_stock_price_without_Mt_MSF()
    plt.plot(t, port_val, label='With Mt', color='blue')
    plt.plot(t, port_val_without, label='Without Mt', linestyle='--', color='red')
    plt.xlabel('Time (Years)')
    plt.ylabel('MSF Portfolio value')
    plt.title(f'MSF Rolling Portfolio Value (Psi = {psi})')
    plt.legend()


# Plot cash account values with and without Mt for all psi values
plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=2*len(psi_values)+1):
    plt.subplot(3, len(psi_values), i)
    _, _, _, _, _, cash_ac_with,_, _, _, _, _, _, _, _, _, _, _, _, _ = simulate_stock_price_with_Mt_LVOL(psi)
    _, cash_ac_without, _, _ = simulate_stock_price_without_Mt_MSF_LVOL()

    plt.plot(t, cash_ac_with, label='With Mt', color='blue')
    plt.plot(t, cash_ac_without, label='Without Mt', linestyle='--', color='red')
    plt.xlabel('Time (Years)')
    plt.ylabel('MSF Portfolio value')
    plt.title(f'Cash Account Value (Psi = {psi})')
    plt.legend()


plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=2*len(psi_values)+1):
    plt.subplot(3, len(psi_values), i)
    S_with, _, _, _, _, _,_, _, _, _, _, _, _, _, _, _, _, _, _ = simulate_stock_price_with_Mt(psi)
    S_without, _, _, _= simulate_stock_price_without_Mt_MSF()
    plt.plot(t, S_with, label='With Mt', color='blue')
    plt.plot(t, S_without, label='Without Mt', linestyle='--', color='red')
    plt.xlabel('Time (Years)')
    plt.ylabel('Stock Price value')
    plt.title(f'Stock Value (Psi = {psi})')
    plt.legend()

plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=2*len(psi_values)+1):
    plt.subplot(3, len(psi_values), i)
    S_with, _, _, _, theta_with, _,_, _, _, _, _, _, _, _, _, _, _, _, _ = simulate_stock_price_with_Mt(psi)
    S_without, _, _, theta_without = simulate_stock_price_without_Mt_MSF()
    plt.plot(t, theta_with, label='With Mt', color='blue')
    plt.plot(t, theta_without, label='Without Mt', linestyle='--', color='red')
    plt.xlabel('Time (Years)')
    plt.ylabel('Number of Stocks')
    plt.title(f'Number of Stocks (Psi = {psi})')
    plt.legend()


plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=2*len(psi_values)+1):
    plt.subplot(3, len(psi_values), i)
    S_with, _, _, _, _, _,_, _, _, _, _, _, _, _, _, _, _, _, _ = simulate_stock_price_with_Mt(psi)
    S_without, _, _, _= simulate_stock_price_without_Mt_MSF()
    plt.plot(t, S_with, label='With Mt', color='blue')
    plt.plot(t, S_without, label='Without Mt', linestyle='--', color='red')
    plt.xlabel('Time (Years)')
    plt.ylabel('Stock Price value')
    plt.title(f'Stock Value (Psi = {psi})')
    plt.legend()

plt.figure(figsize=(18, 12))
for i, psi in enumerate(psi_values, start=2*len(psi_values)+1):
    plt.subplot(3, len(psi_values), i)
    S_with, _, _, _, theta_with, _,_, _, _, _, _, _, _, _, _, _, _, _, _ = simulate_stock_price_with_Mt(psi)
    S_without, _, _, theta_without = simulate_stock_price_without_Mt_MSF()
    plt.plot(t, theta_with, label='With Mt', color='blue')
    plt.plot(t, theta_without, label='Without Mt', linestyle='--', color='red')
    plt.xlabel('Time (Years)')
    plt.ylabel('Number of Stocks')
    plt.title(f'Number of Stocks (Psi = {psi})')
    plt.legend()



    # Ivol and call prices (TRIAL)
    # striky dleta 
import QuantLib as ql
import math
import numpy as np
import matplotlib.pyplot as plt


as_of_date = 0
# Market Data
tenors = [0.25, 0.5, 0.75, 1]
maturities = [0.25, 0.5, 0.75, 1]
atm_volatilities = [0.2] # bsm assumption initial
smiles = [0]

delta_type = 1
rTS = 0
qTS = 0
spot = 100

solver = ql.Brent()
accuracy = 1e-16
step = 1e-12

class TargetFun:
    def __init__(self, as_of_date, spot, rdf, qdf, strike, maturity, deltas, delta_type, atm_vol, atm_type, smile, interp):
        self.ref_date = as_of_date
        self.strike = strikes
        self.maturity = maturity
        self.spot, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = simulate_stock_price_with_Mt(0.4)
        self.rDcf = rdf # discount factor
        self.qDcf = qdf # discount factor
        self.t = t
        self.deltas = black_scholes_delta(spot, strike, rdf, atm_vol, maturity, t)
        self.interp = interp(self.deltas, smile)
        self.delta_type = delta_type

        stdDev = math.sqrt(self.maturity - self.t) * atm_vol
        calc = ql.BlackDeltaCalculator(ql.Option.Put, delta_type, self.spot, self.rDcf, self.qDcf, stdDev)
        k = calc.atmStrike(atm_type)
        calc = ql.BlackDeltaCalculator(ql.Option.Put, ql.DeltaVolQuote.Spot, self.spot, self.rDcf, self.qDcf, stdDev)
        self.deltas.insert(2, calc.deltaFromStrike(k))
        self.smile = smile.copy()
        self.smile.insert(2, atm_vol)
        self.interp = interp(self.deltas, self.smile)
        self.delta_type = ql.DeltaVolQuote.Spot
        
    def __call__(self, v0):
        optionType = ql.Option.Put
        stdDev = math.sqrt(self.t) * v0
        calc = ql.BlackDeltaCalculator(optionType, self.delta_type, self.spot, self.rDcf, self.qDcf, stdDev)
        d = calc.deltaFromStrike(self.strike)
        v = self.interp(d, allowExtrapolation=True)
        return (v - v0)



vol_vs_strike = []
strikes = np.linspace(S0-100, S0+100, 500, endpoint=False)
display_maturity = as_of_date + ql.Period('3M')

def strike2vol(k):
    vol_by_tenor = []
    for i, smile in enumerate(smiles):
        mat = maturities[i]
        target = TargetFun(as_of_date, 
                           spot,
                           rTS.discount(mat), 
                           qTS.discount(mat), 
                           k, 
                           maturities[i], 
                           deltas[i], 
                           delta_type, 
                           atm_volatilities[i], 
                           1, 
                           smile, 
                           ql.LinearInterpolation) # usually use cubic spline (ql.CubicNaturalSpline)
        guess = atm_volatilities[i]
        vol_by_tenor.append(solver.solve(target, accuracy, guess, step))
    return vol_by_tenor

for k in strikes:
    vol_by_tenor = strike2vol(k)
    vts = ql.BlackVarianceCurve(as_of_date, 
                                maturities,
                                vol_by_tenor, 
                                ql.Actual365Fixed(), 
                                False)
    vts.enableExtrapolation()
    vts_handle = ql.BlackVolTermStructureHandle(vts)
    vol_vs_strike.append(vts_handle.blackVol(display_maturity, 1.0)) # the strike 1.0 has no effect

plt.plot(strikes, vol_vs_strike)
plt.xlabel('Strike')
plt.ylabel('Volatility')
plt.title(f'Volatility at 3M ({display_maturity})')
plt.grid(True)
plt.show()