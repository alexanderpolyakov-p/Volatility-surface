import numpy as np
import heston
from scipy import integrate
from scipy.spatial.distance import cdist

class HestonLSV(heston.Heston):

    def __init__(self, parameters, s0, r, qe_params, bins):
        super().__init__(parameters, s0, r, qe_params)
        self.bins = bins # [strikes], [maturities], [local volatilities]
        self.bump = None 

    def euler_scheme(self, N, T, n, seed_):
            
        sigma, kappa, theta, v0, rho = self.params
        dt, self.dt = 1 / n, 1 / n  

        stockpaths = np.zeros((N, T * n + 1))
        volpaths = np.zeros((N, T * n + 1))

        stochastic_component = np.ones((N, n * T + 1)) * v0
        local_component = np.ones((N, n * T + 1)) * v0

        stockpaths[:, 0] = self.s0
        volpaths[:, 0] = v0

        X = np.stack((self.bump[0] / 100,  self.bump[1]), axis = 1) 
        Y = np.array(self.bump[2]) ** 2 

        # Brownian motions
        np.random.seed(seed_)
        dW1 = np.random.normal(0, np.sqrt(dt), (N, n * T))
        dW2 = rho * dW1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), (N, n * T))

        for i in range(1, T * n + 1):

            volpaths[:, i] = (volpaths[:, i - 1] 
                            + kappa * (theta - np.maximum(volpaths[:, i-1], 0.000000000001)) * dt 
                            + sigma * np.sqrt(np.maximum(volpaths[:, i-1], 0.000000000001)) * dW1[:, i-1])
            volpaths[:, i] = np.maximum(volpaths[:, i], 0.00000000001)

            stockpaths[:, i] = (stockpaths[:, i - 1] * 
                                (1 + self.r * dt 
                                + np.sqrt(volpaths[:, i - 1]) 
                                * np.sqrt((local_component[:, i - 1]) / stochastic_component[:, i - 1]) 
                                * dW2[:, i-1]))

            #local component: 
            stock_time_pairs = np.array(list(zip(stockpaths[:, i] / 100, np.ones(N) * (i / n))))
            distances = cdist(X, stock_time_pairs, 'euclidean').T
            min_indices = np.argmin(distances, axis=1)
            local_component[:, i] = Y[min_indices]

            #stochastic component:
            collection = np.digitize(stockpaths[:, i], self.bins, right = False) 

            for g in np.unique(collection):
                indices = np.where(collection == g)
                stochastic_component[:, i][indices] = np.mean([volpaths[:, i][indices]])

        self.paths = stockpaths
        return stockpaths

    def broadie_kaya_scheme(self, N, T, n, seed_):

        n_steps = T * n + 1
        sigma, kappa, theta, v0, rho = self.params 

        dt, self.dt = 1 / n, 1 / n

        stockpaths = np.ones((N, T * n + 1)) * self.s0
        volpaths = np.ones((N, T * n + 1)) * v0
        expected_stochastic_varince = np.ones((N, n * T + 1)) * v0
        local_volatility = np.ones((N, n * T + 1)) * v0

        # Brownian motions
        np.random.seed(seed_)
        dW1 = np.random.normal(0, 1, size = (N, n * T))

        for i in range(1,  n_steps):
            #generate volatility step: 
            cd = (sigma ** 2) / (4 * kappa) * (1 - np.exp(-kappa * dt))
            df = 4 * kappa * theta / sigma **2
            nonc = (4 * kappa * np.exp(- kappa * dt)) / (sigma ** 2 * (1 - np.exp( - kappa * dt))) * volpaths[:, i - 1]
            volpaths[:, i] = cd * np.random.noncentral_chisquare(df, nonc) 

            local_sigma =  np.sqrt(local_volatility[:, i - 1] **2 / expected_stochastic_varince[:, i - 1])
            
            stockpaths[:,i] =   stockpaths[:, i - 1] * \
                                          np.exp(self.r * dt + 
                                                 local_sigma * ((kappa * rho / sigma) - (0.5 * local_sigma)) * volpaths[:, i - 1] * dt + \
                                                 (local_sigma * rho / sigma) * (volpaths[:,i]  - volpaths[:,i - 1] - kappa * theta * dt)
                                                 + local_sigma * np.sqrt((1 - rho**2) * volpaths[:, i] * dt) * dW1[:, i - 1])

            #STOCHASTIC VOLATILITY
            collection = np.digitize(stockpaths[:, i], self.bins, right = False) 

            for g in np.unique(collection):
                indices = np.where(collection == g)

                expected_stochastic_varince[:, i][indices] = np.mean([volpaths[:, i][indices]])

            #LOCAL VOLATILITY
            X = np.stack((self.bump[0] / 100,  self.bump[1]), axis = 1) 
            Y = np.array(self.bump[2])
            stock_time_pairs = np.array(list(zip(stockpaths[:, i] / 100, np.ones(N) * (i / n))))
            min_indices = np.argmin(np.linalg.norm(X[:, None, :] - stock_time_pairs, axis=2).T, axis = 1)
            local_volatility[:, i] = Y[min_indices]
            
        self.paths = stockpaths
        return stockpaths 


    def set_bump(self, grid): 
        
        self.bump = grid