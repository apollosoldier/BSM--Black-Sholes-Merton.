# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 05:42:01 2022
@author: TRAORE Mohamed
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prefect import flow, task
from rich import inspect
from scipy.stats import norm
from europeanOption import EuropeanOption
from typing import Tuple
# Filter out warnings
import warnings
warnings.filterwarnings("ignore")

# Set the default figure size for matplotlib
plt.rcParams["figure.figsize"] = (15, 5)

class Strangle(EuropeanOption):
    def __init__(self, s0: float, r: float, sigma: float, t: float, K1, K2):
        self.K1 = K1
        self.K2 = K2
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.t = t

    def prices(self, num_sims):
        s0_values = np.linspace(self.s0, self.K2, num_sims)
        strangle_prices = np.zeros(len(s0_values))
        call_prices = np.zeros(len(s0_values))
        put_prices = np.zeros(len(s0_values))

        for index, s0 in enumerate(s0_values):
            # Calculate the price of the call and put options
            call_prices[index] = EuropeanOption(s0=s0, k=self.K1, r=r, sigma=self.sigma, t=self.t).call_price()
            put_prices[index] = EuropeanOption(s0=s0, k=self.K2, r=r, sigma=self.sigma, t=self.t).put_price()
            # Calculate the price of the strangle option
            strangle_prices[index] = call_prices[index] + put_prices[index]

        return strangle_prices, call_prices, put_prices, s0_values

    def plot_strangle(self, num_sims: int):
        payoff_strangle, call_prices, put_prices, s0_values = self.prices(num_sims)
        print("Max Profit: Unlimited")
        print("Max Loss:", min(payoff_strangle))
        # Plot
        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)  # Top border removed
        ax.spines["right"].set_visible(False)  # Right border removed
        ax.spines["bottom"].set_position("zero")  # Sets the X-axis in the center

        ax.plot(s0_values, call_prices, "--", label="Call", color="r")
        ax.plot(s0_values, put_prices, "--", label="Put", color="g")

        ax.plot(s0_values, payoff_strangle, label="Strangle")
        ax.set_title(
            f"Strangle with : s0 = {self.s0}, K1 = {self.K1}, K2 = {self.K2}, r = {self.r}, sigma = {self.sigma} and t = {self.t}\nPremium Call = 0 cents / Premium Put = 0 cents"
        )
        plt.xlabel("Stock Price", ha="left")
        plt.ylabel("Profit and loss")
        plt.legend()
        plt.grid()
        plt.show()


class Straddle(EuropeanOption):
    def __init__(self, s0: float, r: float, sigma: float, t: float, K):
        super().__init__(s0, K, r, sigma, t)
        self.tmp = self.s0

    def prices(self, num_sims):
        s0_values = self.dummy_simulation_stock_price(num_sims)
        straddle_prices = np.zeros(len(s0_values))
        call_prices = np.zeros(len(s0_values))
        put_prices = np.zeros(len(s0_values))

        for index, s0 in enumerate(s0_values):
            self._s0 = s0
            call_prices[index] = super().call_price()  # Premium Call = 0 cents
            put_prices[index] = super().put_price()  # Premium Put = 0 cents
            straddle_prices[index] = (
                call_prices[index] + put_prices[index]
            )  # premium à 20 centimes (soit 40 centimes les 2)
        self._s0 = self.tmp

        return straddle_prices, call_prices, put_prices, s0_values

    def plot_straddle(self, num_sims: int):
        payoff_straddle, call_prices, put_prices, s0_values = self.prices(num_sims)
        print("Max Profit: Unlimited")
        print("Max Loss:", min(payoff_straddle))
        # Plot
        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)  # Top border removed
        ax.spines["right"].set_visible(False)  # Right border removed
        ax.spines["bottom"].set_position("zero")  # Sets the X-axis in the center

        ax.plot(s0_values, call_prices, "--", label="Call", color="r")
        ax.plot(s0_values, put_prices, "--", label="Put", color="g")
        ax.set_title(
            f"Straddle with : s0 = {super().s0}, K = {super().k}, r = {super().r}, sigma = {super().sigma} and t = {super().t}\n Call Premium = 0/ Put Premium 0"
        )

        ax.plot(s0_values, payoff_straddle, label="Straddle")
        plt.xlabel("Stock Price", ha="left")
        plt.ylabel("Profit and loss")
        plt.legend()
        plt.grid()
        plt.show()


class Simulation(EuropeanOption):
    def __init__(
        self,
        s0: float,
        k: float,
        r: float,
        sigma: float,
        t: float,
        num_sims: float,
        k2: float,
    ):
        super().__init__(s0, k, r, sigma, t)
        self.num_sims = num_sims
        self.s0 = s0
        self.k = k
        self.r = r
        self.sigma = sigma
        self.t = t
        self.k2 = k2

    # @task(retries=3, retry_delay_seconds=3)
    def exercise1(self):
        s0 = self.s0
        K1 = self.k
        K2 = self.k2
        r = self.r
        sigma = self.sigma
        T = self.t
        num_sims = self.num_sims
        print(
            f"Simulation with: s0 = {self.s0}, K1 = {self.k}, K2 = {self.k2}, r = {self.r}, sigma = {self.sigma} and T = {self.t}"
        )

        print("=================== 1-Implement these results in Python. (call+ put) ===================\n")
        euro_option = EuropeanOption(s0=s0, K1=K1, K2=K2, r=r, sigma=sigma, T=T)
        # Calculate the price of the call option
        option_call_price = super().call_price()
        # Calculate the price of the put option
        option_put_price = super().put_price()

        # Print the option price
        print(f"The price of the call option is {option_call_price}")
        print(f"The price of the put option is {option_put_price}\n")

        print("=================== 2-Tracer sous Python ces résultats (en faisant varier S) ===================\n")
        s0_values = euro_option.dummy_simulation_stock_price(num_sims)
        call_prices = np.zeros(len(s0_values))
        put_prices = np.zeros(len(s0_values))
        for index, s0 in enumerate(s0_values):
            euro_option = EuropeanOption(s0=s0, K1=K1, K2=K2, r=r, sigma=sigma, T=T)
            # Calculate the price of the call and put
            call_prices[index] = euro_option.call_price()
            put_prices[index] = euro_option.put_price()

        plt.plot(s0_values, call_prices, label="Call", color="r")
        plt.plot(s0_values, put_prices, label="Put", color="g")
        plt.title(
            f"Plot Call & Put with S = [{min(s0_values), max(s0_values)}]\nPremium Call = 0 cents / Premium Put = 0 cents"
        )
        plt.ylabel("Profit and loss")
        plt.legend(["Call Option Price", "Put Option Price"])
        plt.grid()
        plt.show()
        print("=======================================================================")


        print(
            "3-Vérifier la formule de parité Call-Put. (Retrouver la formule vue en cours et coder une fonction qui la vérifie)"
        )
        ##### Parity
        parities = []
        for index, s0 in enumerate(s0_values):
            euro_option.s0 = s0
            # Parity estimation
            parities.append(euro_option.parity())
        euro_option.s0 = tmp

        if parities.count(False) > 0:
            print("Parity is NOK")
        else:
            print("Parity is OK !")
        print("=======================================================================")

        print("=================== 4 En déduire la valeur du delta ===================")
        delta_call = np.exp(-self.r * self.t) * norm.cdf(euro_option.d1())

        print("Delta call = ", delta_call, " \nDelta put = ", delta_call - 1)
        print("=======================================================================")

        print(
            "5. Implémenter un code qui vous retourne les valeurs delta pour le call et pour le put, Tracer les delta_call et delta_put"
        )
        print(
            "Delta call : self.delta_call()", "\nDelta put : self.delta_put.__class__()"
        )
        delta_call = np.zeros(len(s0_values))
        delta_put = np.zeros(len(s0_values))
        for index, s0 in enumerate(s0_values):
            euro_option.s0 = s0
            # Parity estimation
            delta_call[index] = euro_option.delta_call()
            delta_put[index] = euro_option.delta_put()
        euro_option.s0 = tmp
        print("=================== Plot deltas ===================")
        plt.plot(delta_call, label="Delta call")
        plt.plot(delta_put, label="Delta put")
        plt.title(f"Delta call & Delta put")
        plt.legend()
        plt.grid()
        plt.show()
        print("=======================================================================")
        print(
            "6. Pricer le prix d’un strangle et d’un straddle avec les formules Black Scholes, implémenter et tracer graphiquement sous Python."
        )
        s0 = self.s0
        K1 = self.k
        K2 = self.k2
        r = self.r
        sigma = self.sigma
        t = self.t
        print(
            f"Straddle with : s0 = {s0}, K = {K1}, r = {r}, sigma = {sigma} and t = {t}"
        )
        straddle = Straddle(s0=s0, r=r, sigma=sigma, t=t, K=K1)
        straddle.plot_straddle(num_sims=self.num_sims)
        print(
            f"Strangle with : s0 = {s0}, K1 = {K1}, K2 = {K2}, r = {r}, sigma = {sigma} and t = {t}"
        )
        strangle = Strangle(s0=s0, r=r, sigma=sigma, t=t, K1=K1, K2=K2)
        strangle.plot_strangle(num_sims=self.num_sims)


class InterestRateSwap:
    def __init__(self, T: float, spread: float, fixed_rate: float, nominal: float):
        self.T = T
        self.fixed_rate = fixed_rate
        self.spread = spread
        self.nominal = nominal
        self.result = pd.DataFrame(
            columns=[
                "T",
                "pv_fixed_leg",
                "variable_rate",
                "pv_variable_leg",
                "variable_leg_cf",
                "risk_free_rate",
                "swap_price",
            ]
        )

    def risk_free_rate(self, t):
        """
        Computes the risk-free rate at time t.
        """
        return self.spread * t

    def variable_rate(self, t):
        """
        Computes the variable rate at time t.
        """
        return self.spread * t + self.spread

    def swap_price(self):
        """
        Prices the swap by computing the present value of the fixed and variable leg cash flows.
        """
        pv_fixed_leg = 0
        pv_variable_leg = 0
        for t in range(1, self.T + 1):
            fixed_leg_cf = self.nominal * self.fixed_rate
            pv_fixed_leg += fixed_leg_cf / (1 + self.risk_free_rate(t)) ** t

            variable_leg_cf = self.nominal * self.variable_rate(t)
            pv_variable_leg += variable_leg_cf / (1 + self.risk_free_rate(t)) ** t
            self.result = self.result.append(
                {
                    "T": t,
                    "pv_fixed_leg": pv_fixed_leg,
                    "variable_rate": self.variable_rate(t),
                    "pv_variable_leg": pv_variable_leg,
                    "variable_leg_cf": variable_leg_cf,
                    "risk_free_rate": self.risk_free_rate(t),
                    "swap_price": pv_fixed_leg - pv_variable_leg,
                },
                ignore_index=True,
            )
        swap_price = pv_fixed_leg - pv_variable_leg
        return swap_price, self.result


# @flow
def exercice1(S0, K1, K2, r, sigma, T, N_SIMULATION):
    print("=================== EXERCICE 1 ===================")
    new_instance = Simulation(
        s0=S0, k=K1, k2=K2, r=r, sigma=sigma, t=T, num_sims=N_SIMULATION
    )
    new_instance.exercise1()
    print("==================================================")


def excercise2(T, spread, fixed_rate, nominal):
    print("=================== EXERCICE 2 ===================")
    itrw = InterestRateSwap(T=T, spread=spread, fixed_rate=fixed_rate, nominal=nominal)
    sw_price, result = itrw.swap_price()
    print(inspect(result))
    print(inspect(f"The price of the swap is: {sw_price:.2f} €"))
    print("==================================================")


def exercise3(T: int, taux_fixe: float, nominal: float) -> Tuple[pd.DataFrame, float]:
    print("=================== EXERCISE 3 ===================")
    value_trs = 0
    result = pd.DataFrame(
        columns=[
            "T",
            "index_performance",
            "risk-free_rate",
            "value_trs",
            "fixed_rate",
        ]
    )
    for t in range(1, T + 1):
        index_performance = (2 / 100) * t
        risk_free_rate = (1 / 100) * t
        value_trs += (index_performance - risk_free_rate) * nominal
        result = result.append(
            {
                "T": t,
                "index_performance": index_performance,
                "risk-free_rate": risk_free_rate,
                "value_trs": value_trs,
                "fixed_rate": taux_fixe,
            },
            ignore_index=True,
        )
    value_trs += taux_fixe * nominal
    print(inspect((result)))
    print(f"Valuation of S&P 500 is equal to: {round(value_trs, 3)} €")
    print("==================================================")
    return result, value_trs



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-P", "--underlying_asset", type=int, help="the underlying asset price"
    )
    parser.add_argument(
        "-K1", "--strike_price_1", type=int, help="strike price for call option 1"
    )
    parser.add_argument(
        "-K2", "--strike_price_2", type=int, help="strike price for call option 2"
    )
    parser.add_argument("-r", "--rate", type=float, help="the risk-free rate")
    parser.add_argument("-T", "--duration", type=int, help="the duration of the option")
    parser.add_argument(
        "-n", "--num_simulations", type=int, help="the number of simulations to run"
    )
    parser.add_argument(
        "-s", "--sigma", type=float, help="the volatility of the underlying asset"
    )

    args = parser.parse_args()
    if all(v is not None for v in vars(args).values()):
        S0 = args.underlying_asset
        K1 = args.strike_price_1
        K2 = args.strike_price_2
        r = args.rate
        sigma = args.sigma
        T = args.duration
        N_SIMULATION = args.num_simulations
        exercice1(S0, K1, K2, r, sigma, T, N_SIMULATION)
    else:
        print("Please provide arguments for exercise 1")
        print(parser.print_help())

    T = 5
    spread = 0.01  # 1%
    fixed_rate = 0.04  # 4%
    nominal = 1_000_000
    excercise2(T, spread, fixed_rate, nominal)

    taux_fixe = 0.03
    performance_annuelle = 0.02
    excercice3(T, taux_fixe, nominal)
