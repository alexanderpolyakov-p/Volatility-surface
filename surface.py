import grid
import numpy as np
import pandas as pd
import basic_functions 

from scipy.interpolate import RectBivariateSpline
from scipy.spatial.distance import cdist

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


class local_polynomial_regression_2D: 

    def __init__(self, kernel_size, kernel_type):
    
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        self.polynomial = make_pipeline(PolynomialFeatures(2), LinearRegression())

    def fit(self, strike, tau, iv):
        X = np.stack((np.array(strike), np.array(tau)), axis=1)
        Y = np.array(iv)
        self.X, self.Y = X, Y 
        return self

    def kernel(self, distance): 
        if self.kernel_type == 'gaussian':
            return np.exp(-0.5 * ((distance) / self.kernel_size) ** 2)
        
        elif self.kernel_type == 'epanechnikov':
            u = (distance) / self.kernel_size
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

    def predict(self, strike, tau):
        X0I, X1I = np.meshgrid(strike, tau)
        X_predict = np.vstack((np.atleast_2d(X0I.flatten()), np.atleast_2d(X1I.flatten()))).T

        predictions = np.array([])
        for xi in X_predict: #xi = [s, k]
            distance = cdist(self.X.reshape(-1, 2), [xi], 'euclidean').flatten()
            weights_at_point = self.kernel(distance)
            self.polynomial.fit(self.X.reshape(-1, 2), self.Y, linearregression__sample_weight = weights_at_point)
            predictions = np.append(predictions, self.polynomial.predict(np.atleast_2d(xi)))

        res = pd.DataFrame(X_predict, columns = ['strike', 'maturity'])
        res['iv'] = predictions
        return res
    

class Surface:

    def __init__(self, grid, r): 

        self.initial_grid = grid 
        self.implied_total_variance = (grid.iv**2) * grid.maturities
        self.forward_log_moneyness = np.log(grid.strikes/ (grid.s0 * np.exp(r * grid.maturities)))
        self.log_moneyness = None
        self.r = r

        #DERIVATIVES: 
        self.interpolant = None 

        dV_dt = None 
        dV_dlogK = None 
        d2V_dlogK2 = None 

        dC_dt = None 
        dC_dK = None 
        d2C_dK2 = None 

    def calculate_iv(self): 
        iv = np.zeros(len(self.strikes))
        for ci in range(len(self.prices)):
            iv[ci] = basic_functions.implied_volatility(self.prices[ci], 
                                                        self.initial_grid.s0, 
                                                        self.maturities[ci], 
                                                        self.strikes[ci], 
                                                        self.r, 'call')
        self.implied_volatility = iv 

    def set_interpolant(self, itype, field, band):
        
        if field == 'price': 
            z = self.initial_grid.prices
        elif field == 'iv': 
            z = self.implied_total_variance

        #SET LOCAL POLYNOMIAL INTRPOLANT
        if itype =='local_polynomial':
            polynomial_iterpolant = local_polynomial_regression_2D(band, 'epanechnikov')
            polynomial_iterpolant.fit(self.grid.strikes, self.grid.maturities, z)
            self.interpolant = polynomial_iterpolant.predict

            # self.dV_dt = 
            # self.dV_dlogK = 
            # self.d2V_dlogK2 = 

        #SET SPLINE INTRPOLANT
        elif itype =='splines': 
            x = np.unique(self.initial_grid.strikes)
            y = np.unique(self.initial_grid.maturities)
            X, Y = np.meshgrid(x, y)
            Z = z.reshape(7,9)
            interp_func = RectBivariateSpline(y, x, Z)
            self.iterpolant = interp_func

            if field == 'iv':
                self.dV_dt = interp_func.partial_derivative(dx=1, dy = 0)
                self.dV_dlogK = interp_func.partial_derivative(dx = 0, dy = 1)
                self.d2V_dlogK2 = interp_func.partial_derivative(dx = 0, dy = 2)

            if field == 'price':
                self.dC_dt = interp_func.partial_derivative(dx=1, dy = 0)
                self.dC_dK = interp_func.partial_derivative(dx = 0, dy = 1)
                self.d2C_dK2 = interp_func.partial_derivative(dx = 0, dy = 2)

    def implied_volatility(self, K, T):
        k = np.log(K/ (self.initial_grid.s0 * np.exp(self.r * T)))
        w = self.iterpolant(T, K, grid = False)
        return np.sqrt(w/T)

    def local_volatility(self, K, T, field): 
        
        if field == 'iv':
            k = np.log(K/ (self.initial_grid.s0 * np.exp(self.r * T)))

            local_vol_squared = self.dV_dt(T, K, grid = False) / \
                                (1 - k / self.iterpolant(T, K, grid = False) * 
                                self.dV_dlogK(T, K, grid = False) +
                                0.25 * (-0.25 - 1 / self.iterpolant(T, K, grid = False) + 
                                k * k / (self.iterpolant(T, K, grid=False)**2)) * 
                                self.dV_dlogK(T, K, grid = False) **2 +
                                0.5 * self.d2V_dlogK2(T, K, grid = False))

            return np.sqrt(local_vol_squared)
    
        elif field == 'price': 
            local_vol_squared = (self.dC_dt(T, K, grid = False)  + self.r * K * self.dC_dK(T, K, grid = False)) / \
                                (0.5 * K * K * self.d2C_dK2(T, K, grid = False))

            return np.sqrt(local_vol_squared)

    def svi(self, predict_strikes, predict_maturities):    

        def SVI_function(params, strikes):
            a, b, rho, m, sigma = params
            return a + b * (rho * (strikes - m) + np.sqrt((strikes - m) ** 2 + sigma ** 2))

        def objective_func(params, strikes, implied_vols, maturity):
            totalvols = implied_vols * maturity 
            model_vols = self.SVI_function(params, strikes)
            return np.sum((model_vols - totalvols) ** 2)
      
