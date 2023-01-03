from scipy.stats import norm
import numpy as np
from model import BlackScholes


class EuropeanOption(BlackScholes):
    def __init__(self, s0: float, k: float, r: float, sigma: float, t: float):
        # Initialize class with parameters
        super().__init__(s0, k, r, sigma, t)

        self.s0 = s0
        self.k = k
        self.r = r
        self.sigma = sigma
        self.t = t

    @property
    def s0(self):
        return self._s0

    @s0.setter
    def s0(self, value):
        if value <= 0:
            raise ValueError("S0 must be positive")
        self._s0 = value

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        if value <= 0:
            raise ValueError("K must be positive")
        self._k = value

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        if value <= 0:
            raise ValueError("T must be positive")
        self._t = value

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        if value <= 0:
            raise ValueError("r must be positive")
        self._r = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value <= 0:
            raise ValueError("sigma must be positive")
        self._sigma = value

    def call_price(self):
        call_price = (self.s0 * norm.cdf(super().d1())) - (
            self.k * np.exp(-self.r * self.t) * norm.cdf(super().d2())
        )
        return call_price  # j'ai choisi un premium de 50 centimes pour le call

    def put_price(self):
        put_price = (self.k * np.exp(-self.r * self.t) * norm.cdf(-super().d2())) - (
            self.s0 * norm.cdf(-super().d1())
        )
        return put_price  # j'ai choisi un premium de 80 centimes pour le call

    def geometric_brownian_motion(self, N, mu):
        dt = self.t / N  # time step
        S = np.empty(N)  # array to store the sample path
        S[0] = self.s0  # set the initial price
        S[1:] = [
            S[i - 1]
            * np.exp(
                (mu - 0.5 * self.sigma**2) * dt
                + self.sigma * np.sqrt(dt) * np.random.normal()
            )
            for i in range(1, N)
        ]
        for t in range(1, N):
            S[t] = S[t - 1] * np.exp(
                (mu - 0.5 * self.sigma**2) * dt
                + self.sigma * np.sqrt(dt) * np.random.normal()
            )
        return S

    def dummy_simulation_stock_price(self, num_sims: int) -> np.array:
        return np.linspace(self.s0, self.k, num_sims)

    def parity(self):
        diff = self.call_price() - self.put_price()
        check = self.s0 - self.k * np.exp(-self.r * self.t)
        if not np.isclose([diff], [check]):
            print("Parity is not close to 0.")
            print(
                f"call_price - put_price) = {diff}  -------------  S0 + K * np.exp(-r * T) = {check}"
            )
            return False
        return True

    def delta_call(self):
        d1 = (np.log(self.s0 / self.k) + (self.r + self.sigma**2 / 2) * self.t) / (
            self.sigma * np.sqrt(self.t)
        )
        delta_call = np.exp(-self.r * self.t) * norm.cdf(d1)
        return delta_call

    def delta_put(self):
        delta_put = self.delta_call() - 1
        return delta_put