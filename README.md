
### Volatility-Surface
#### Grid (grid.py)
The Grid object holds general market information, including the current stock price, strikes, maturities of options, and their prices. It also provides functionality to calculate the implied volatility across the entire grid at given strikes and maturities.

#### Volatility Surface (surface.py)
The Volatility Surface object focuses on constructing the volatility surface, utilizing parameters from the Grid object. It includes methods for interpolating implied volatility points using splines and 2D kernel local polynomial regression. Additionally, it estimates the shape of the local volatility surface at specified points using the Dupire and Gatheral approach. Once constructed, the methods of the Volatility Surface can be used as instrumental functions during simulations.

#### Stochastic Model (stochastic_model.py)
This base object facilitates the development of stochastic volatility models. It incorporates market information and offers methods for pricing using both analytic and Monte Carlo methods.

#### Heston Model (heston.py)
Derived from stochastic_model.py, this object specifies the Heston stochastic volatility model. It includes two types of analytic pricing—using the original paper and Lewis formula—and three types of Monte Carlo simulation schemes: Euler method, Andersen scheme, and Broadie-Kaya scheme.

#### Heston LSV Model (heston_lsv.py)
Derived from heston.py, this model adds an additional local structure to volatility during simulation. The local volatility surface is encoded into the new Monte Carlo schemes as a bump variable. The MC schemes include simple Euler simulation and the Broadie-Kaya scheme.
