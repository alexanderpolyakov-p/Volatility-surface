import numpy as np
from scipy import integrate
import basic_functions
import stochastic_model

vector_norminvcdf = np.vectorize(basic_functions.point_norminvcdf)

class Heston(stochastic_model.StochasticModel):

    def __init__(self, parameters, s0, r, qe_params):
        super().__init__(parameters, s0, r)
        self.qe_params = qe_params

    '''
    Analytic 
    '''

    def pricer(self, tau, K, flag):
        
        sigma, kappa, theta, V0, rho = self.params

        if flag == 'heston':
        
            def Probability(_1_=True):
                
                if _1_:
                    u = 0.5
                    b = kappa - (rho * sigma)
                else:
                    u = -0.5
                    b = kappa
                return (1/2) + (1/np.pi) * real_integral(u, b)

            def integrand(phi, u, b):
                return np.real((np.exp(-1 * 1j * phi * np.log(K)) * heston_characteristic(u, b, phi)) / (1j * phi))

            def real_integral(u, b):
                integral_func = lambda phi: integrand(phi, u, b)
                return integrate.quad(integral_func, 0, 100)[0]

            def heston_characteristic(u, b, phi):

                d = np.sqrt((rho * sigma * phi * 1j - b)**2 - (sigma**2) * (2 * u * phi * 1j - (phi**2)))
                g = (b - (rho * sigma * phi * 1j) + d) / (b - (rho * sigma * phi * 1j) - d)
                a = kappa * theta

                C = self.r * phi * 1j * tau + (a/(sigma**2)) * ((b - (rho * sigma * phi * 1j) + d) * tau - 2 * np.log((1 - g * np.exp(d * tau)) / (1 - g)))
                D = ((b - (rho * sigma * phi * 1j) + d) / (sigma**2)) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))

                return np.exp(C + (D * V0) + 1j * phi * np.log(self.s0))
            
            return self.s0 * Probability(True) - K * np.exp(-self.r * tau) * Probability(False)

        elif flag == 'lewis':

            V = sigma ** 2

            def f(k_in):
                k = k_in + 0.5 * 1j
                b = (2.0 / V) * (1j * k * rho * sigma + kappa)
                e = np.sqrt(b**2 + 4.0 * k * (k - 1j) / V)
                g = (b - e) / 2.0
                h = (b - e) / (b + e)
                q = V * tau / 2.0
                Q = np.exp(-e * q)
                H = np.exp((2.0 * kappa * theta / V) * (q * g - np.log((1.0 - h * Q) / (1.0 - h))) + V0 * g * (1.0 - Q) / (1.0 - h * Q))
                integrand = H * np.exp(-1j * k * X) / (k * k - 1j * k)
                return integrand.real

            F = self.s0 * np.exp(self.r * tau)
            X = np.log(F / K)
            integral = integrate.quad(f, 0.0, np.inf)[0] * (1.0 / np.pi)
            price = self.s0 - K * np.exp(-self.r * tau) * integral

            return price
        
        elif flag == 'gatheral': 

            return 0
    
    '''
    Monte-Carlo simulations
    '''
    
    def euler_scheme(self, N, T, n, seed_):

        sigma, kappa, theta, v0, rho = self.params
        dt, self.dt = 1 / n, 1 / n

        stockpaths = np.ones((N, T * n + 1)) * self.s0
        volpaths = np.ones((N, T * n + 1)) * v0

        np.random.seed(seed_)
        dW1 = np.random.normal(0, np.sqrt(dt), (N, n * T))
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), (N, n * T))

        for i in range(1, T * n + 1):
            volpaths[:, i] = np.abs(volpaths[:, i-1] + kappa * (theta - volpaths[:, i-1]) * dt +
                            sigma * np.sqrt(volpaths[:, i-1]) * dW1[:, i-1])
            
            stockpaths[:, i] = (stockpaths[:, i - 1] * (1 + self.r * dt + np.sqrt(volpaths[:, i - 1]) * dW2[:, i - 1]))

        self.paths = stockpaths
        return stockpaths , volpaths

    def qe_a_scheme(self, N, T, n, seed_): 
        # Due to Leif Andersen(2006)

        n_steps = int(T * n)
        sigma, kappa, theta, v0, rho = self.params 
        sigma2 = sigma * sigma
        psib, gamma1, gamma2 =  self.qe_params
        dt, self.dt = 1 / n, 1 / n

        K0 = np.ones(N) * (-rho * kappa * theta * dt / sigma)

        K1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
        K2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
        K3 = (1 - rho**2) * gamma1 * dt
        K4 = (1 - rho**2) * gamma2 * dt
        A = K2 + 0.5 * K4

        volpaths = np.ones((N, T * n)) * v0
        stockpaths = np.ones((N, T * n)) * self.s0

        np.random.seed(seed_)
        Zv = np.random.normal(size = (N, T * n))
        Zs = rho * Zv + np.sqrt(1.0 - rho * rho) * np.random.normal(size = (N, T * n))
        u = np.random.uniform(size = (N, T * n))

        E = np.exp(-kappa * dt)
        c1 = sigma2 * E * (1.0 - E) / kappa
        c2 = theta * sigma2 * ((1.0 - E)**2) / 2.0 / kappa

        for i in range(1, n_steps): 
            m = theta + (volpaths[:,i-1] - theta) * E
            m2 = m * m
            s2 = c1 * volpaths[:, i - 1] + c2
            psi = s2 / m2

            index_norm = np.where(psi <= psib)
            index_delta = np.where(psi > psib)  

            b2 = 2.0 / psi[index_norm] - 1.0 + np.sqrt((2.0 / psi[index_norm]) * (2.0 / psi[index_norm] - 1.0))
            b = np.sqrt(b2)
            a = m[index_norm] / (1.0 + b2)

            us = u[:, i - 1][index_norm]
            Zv_inverse = vector_norminvcdf(us)

            volpaths[:,i][index_norm] = a * ((b + Zv_inverse)**2)
            M1 = np.exp((A * b2 * a) / (1.0 - 2.0 * A * a)) / np.sqrt(1.0 - 2.0 * A * a)

            p = (psi - 1) / (psi + 1)
            beta = (1.0 - p) / m

            index_uniform = np.where(u[:, i - 1] > p)
            index_inversepdf = np.intersect1d(index_delta, index_uniform)
            index_zero = np.setdiff1d(index_delta,  index_uniform)

            volpaths[:,i][index_inversepdf] = np.log((1.0 - p[index_inversepdf]) / (1.0 - u[:,i -  1][index_inversepdf])) / beta[index_inversepdf]
            volpaths[:,i][index_zero] = 0.0

            M2 = p[index_delta] + (beta[index_delta] * (1 - p[index_delta])) / (beta[index_delta] - A)

            K0[index_norm] = - np.log(M1) - (K1 + 0.5 * K3) * volpaths[:,i - 1][index_norm]
            K0[index_delta]  = - np.log(M2) - (K1 + 0.5 * K3) * volpaths[:,i - 1][index_delta]
            
            stockpaths[:,i] = stockpaths[:, i - 1] * np.exp((self.r * dt + K0 + (K1 * volpaths[:,i-1]) + (K2 * volpaths[:,i]) + np.sqrt(K3 * volpaths[:,i] + K4 * volpaths[:,i-1]) * Zs[:,i - 1])) 

        self.paths = stockpaths
        return stockpaths

    def broadie_kaya_scheme(self, N, T, n, seed_): 

        n_steps = T * n + 1
        sigma, kappa, theta, v0, rho = self.params 
        dt, self.dt = 1 / n, 1 / n

        stockpaths = np.ones((N, T * n + 1)) * self.s0
        volpaths = np.ones((N, T * n + 1)) * v0  

        # Brownian motions
        np.random.seed(seed_)
        dW1 = np.random.normal(0, 1, size = (N, n * T))

        for i in range(1,  n_steps):
   
              cd = (sigma**2) / (4 * kappa) * (1 - np.exp(-kappa * dt))
              df = 4 * kappa * theta / (sigma**2)
              nonc = (4 * kappa * np.exp(- kappa * dt)) / (sigma ** 2 * (1 - np.exp( - kappa * dt))) * volpaths[:, i - 1]
              volpaths[:,i] = cd * np.random.noncentral_chisquare(df, nonc) 

              stockpaths[:,i] =   stockpaths[:, i - 1] * \
                                          np.exp(self.r * dt + 
                                                 ((kappa * rho / sigma) - (0.5)) * volpaths[:, i - 1] * dt + \
                                                 (rho / sigma) * (volpaths[:,i]  - volpaths[:,i - 1] - kappa * theta * dt)
                                                 + np.sqrt((1 - rho**2) * volpaths[:, i] * dt) * dW1[:, i - 1])
              
        self.paths = stockpaths
        return stockpaths
    