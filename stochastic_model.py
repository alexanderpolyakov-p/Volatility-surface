import numpy as np

class StochasticModel(): 

    def __init__(self, parameters, s0, r): 
        self.params = parameters
        self.s0 = s0
        self.r = r
        self.paths = None
        self.dt = None

    def pricer(self, tau, K, flag):
        
        pass

    def analytic_price(self, strikes, maturities, is_call, flag): #include pricer.

        prices = np.zeros(len(strikes))

        for i in range(0, len(strikes)): 

            if is_call == True:
                prices[i] = self.pricer(maturities[i], strikes[i], flag)
            else: 
                prices[i] = self.pricer(maturities[i], strikes[i], flag)

        return prices
    

    def mc_price(self, strikes, maturities, is_call): #include schemes.

        prices = np.zeros(len(strikes))
        time = np.linspace(0, len(self.paths[0] - 1) * self.dt, len(self.paths[0]))

        for i in range(0, len(strikes)): 
            time_index = np.argmin(np.abs(time - maturities[i]))
            s = self.paths[:, time_index]

            if is_call == True:
                prices[i] = np.mean(np.maximum(0.0, s - strikes[i])) * np.exp(-self.r * time[time_index])
            else: 
                prices[i] = np.mean(np.maximum(0.0, strikes[i] - s)) * np.exp(-self.r * time[time_index])

        return prices

    def reset_params(self, params):
         
        self.params = params
        self.paths = None
