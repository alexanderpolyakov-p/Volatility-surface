import numpy as np
import numba as nb
from math import fabs, erf, erfc

'''
Basic calculations
'''
# ***************************************************************************************
# *    Title: Speeding up CDF in Python with Numba
# *    Author: Saeed Amen
# *    Availability: https://github.com/cuemacro/teaching/blob/master/pythoncourse/notebooks/numba_example.ipynb

@nb.njit(cache=True, fastmath=True)
def ndtr_numba(a):

    NPY_SQRT1_2 = 1.0/ np.sqrt(2)

    if (np.isnan(a)):
        return np.nan

    x = a * NPY_SQRT1_2
    z = fabs(x)

    if (z < NPY_SQRT1_2):
        y = 0.5 + 0.5 * erf(x)

    else:
        y = 0.5 * erfc(z)

        if (x > 0):
            y = 1.0 - y

    return y

# ***************************************************************************************

#@nb.njit(fastmath=True, cache=True)
# ***************************************************************************************
# *    Title: inverse Normal CDF
# *    Author: John Herrero
# *    Availability: http:#home.online.no/~pjacklam/notes/invnorm/

def point_norminvcdf(p):

    # Define coefficients in rational approximations
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    c1 = -7.78489400243029E-03
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    d1 = 7.78469570904146E-03
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742

    inverse_cdf = 0.0

    # Define break-points
    p_low = 0.02425
    p_high = 1.0 - p_low

    # If argument out of bounds, raise erro

    if p == 0.0:
        p = 1e-10

    if p == 1.0:
        p = 1.0 - 1e-10

    if p < p_low:
        # Rational approximation for lower region
        q = np.sqrt(-2.0 * np.log(p))
        inverse_cdf = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5)
                       * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q
                                    + 1.0)
    elif p <= p_high:
        # Rational approximation for lower region
        q = p - 0.5
        r = q * q
        inverse_cdf = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * \
            q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    elif p < 1.0:
        # Rational approximation for upper region
        q = np.sqrt(-2.0 * np.log(1 - p))
        inverse_cdf = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5)
                        * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return inverse_cdf
# ***************************************************************************************

'''
Black sholes related.
'''
#Black Sholes formula
@nb.njit(cache = True, fastmath = True)
def black_scholes_price(S, tau, K, r, sigma, option):

    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * tau) * (1.0/(sigma * np.sqrt(tau)))
    d2 = d1 - (sigma * np.sqrt(tau))

    if option == 'call':
        return ndtr_numba(d1) * S - ndtr_numba(d2) * K * np.exp(-r * tau)
    else:
        return -ndtr_numba(-d1) * S + ndtr_numba(-d2) * K * np.exp(-r * tau)

#Black Sholes vega
@nb.njit(cache = True, fastmath = True)
def black_scholes_vega(S, tau, K, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
    return S * np.sqrt(tau) * np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)

#Black Sholes delta
@nb.njit(cache = True, fastmath = True)
def black_scholes_delta(S, tau, K, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
    return ndtr_numba(d1)

#Implied volaitlity using the combination of Newton and Bisection method
@nb.njit(cache = True, fastmath = True)
def implied_volatility(price, S, tau, K, r, option): 
    # set general parameters:
    tolerance = 0.00001
    sigma = 0.2  # initial guess

    # switching rule:
    lower_bound = 0.00001
    upper_bound = 1.9

    delta_function = price - black_scholes_price(S, tau, K, r, sigma, option)
    vega = black_scholes_vega(S, tau, K, r, sigma)
    
    while np.abs(delta_function) > tolerance: 
        
        # Newton Raphson method if sigma is inside bounds, and derivative is not too low:
        if vega > 0.001 and lower_bound < sigma < upper_bound:
            sigma = sigma - delta_function / (-vega)

        # Switch to bisection method if sigma is outside bounds or vega is too low:
        else: 
            if black_scholes_price(S, tau, K, r, sigma, option) > price:
                upper_bound = sigma
            else:
                lower_bound = sigma
            sigma = (upper_bound + lower_bound) / 2

        # update delta_function and vega:
        delta_function = price - black_scholes_price(S, tau, K, r, sigma, option)
        vega = black_scholes_vega(S, tau, K, r, sigma)

    return sigma