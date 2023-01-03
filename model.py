import numpy as np
class BlackScholes:
    def __init__(self, s0: float, k: float, r: float, sigma: float, t: float):
        """
        Initializes the class with the following parameters:
          S0: The initial stock price.
          K: The strike price.
          r: The risk-free interest rate.
          sigma: The volatility of the stock price.
          t: The time to expiration of the option (in years).
        """
        self.s0 = s0
        self.k = k
        self.r = r
        self.sigma = sigma
        self.t = t

    def d1(self):
        # Calculate d1 term in Black-Scholes formula
        d1 = (np.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma**2) * self.t) / (
            self.sigma * np.sqrt(self.t)
        )
        return d1

    def d2(self):
        # Calculate d2 term in Black-Scholes formula
        d2 = self.d1() - self.sigma * np.sqrt(self.t)

        return d2