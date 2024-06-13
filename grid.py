from typing import List
from dataclasses import dataclass
import numpy as np
import basic_functions
import matplotlib.pyplot as plt

class Grid:
    def __init__(self):
        self.s0 = None
        self.strikes = None 
        self.maturities = None 
        self.prices_call = None  
        self.prices_put = None  
        self.iv = None

    def fixed_grid_generate(self, strikes, maturities, s0):
        self.s0 = s0
        strikes = strikes * self.s0
        strikes, maturities = np.meshgrid(strikes, maturities)
        strikes, maturities = strikes.ravel(), maturities.ravel()
        condition = np.logical_and(strikes < self.s0 * (1 + np.sqrt(maturities)),
                                strikes > self.s0 * (1 - np.sqrt(maturities)))
        strikes = strikes[condition]
        maturities = maturities[condition]

        self.strikes = strikes
        self.maturities = maturities
    
    def calculate_iv(self, r):
        iv = np.zeros(len(self.strikes))
        for ci in range(len(self.prices_call)):

            if self.strikes[ci] > self.s0 * np.exp(r  * self.maturities[ci]):
                iv[ci] = basic_functions.implied_volatility(self.prices_call[ci], self.s0, self.maturities[ci], self.strikes[ci], r, 'call')
            else: 
                iv[ci] = basic_functions.implied_volatility(self.prices_put[ci], self.s0, self.maturities[ci], self.strikes[ci], r, 'put')

        self.iv = iv 

    def plot_grid(self, field): 
        fig = plt.figure(figsize=(8, 4), dpi = 100)

        if field == 'price_call': 
            for i in np.unique(self.maturities): 
                indexes = np.where(self.maturities == i)
                x = self.strikes[indexes]
                y = self.prices_call[indexes]
                plt.plot(x,y, 'o-', label = str(i), color = (0, (np.exp(-i)) / 2, (np.exp(-i))), linewidth=0.5)
        
        elif field == 'price_put':
            for i in np.unique(self.maturities): 
                indexes = np.where(self.maturities == i)
                x = self.strikes[indexes]
                y = self.prices_put[indexes]
                plt.plot(x,y, 'o-', label = str(i), color = (0, (np.exp(-i)) / 2, (np.exp(-i))), linewidth=0.5)

        elif field == 'iv': 
            for i in np.unique(self.maturities): 
                indexes = np.where(self.maturities == i)
                x = self.strikes[indexes]
                y =  self.iv[indexes]
                plt.plot(x,y, 'o-', label = str(i), linewidth=0.5)

        plt.legend()
