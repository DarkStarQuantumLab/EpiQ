from scipy.integrate import odeint
import numpy as np
from typing import Tuple, Any
from .base_epidimiology_model_class import EpidemiologyModel

class SIRD(EpidemiologyModel):
    """
        The SIRD Class (Susceptible Infected Recovered Dead) invokes implicitly by 
        epidemiology and optimization classes.
    """

    # def __init__(self, i, S0, I0, R0, D0, N, t, beta, gamma, rho, Tr_matrix, total_cities, mainDf):
    #     self.S0 = S0
    #     self.I0 = I0
    #     self.R0 = R0
    #     self.D0 = D0
    #     self.N = N
    #     self.t = t
    #     self.i = i
    #     self.Tr_matrix = Tr_matrix

    #     if mainDf is not None:
    #         self.total_cities = len(mainDf)
    #     else:
    #         self.total_cities=total_cities


    #     if isinstance(beta, list) and isinstance(gamma, list) and isinstance(rho, list):
    #         for i in range(len(beta)):
    #             self.beta = beta[i]
    #             self.gamma = gamma[i]
    #             self.rho = rho[i]

    #             self.evolve()

    #     if not isinstance(beta, list) and not isinstance(gamma, list) and not isinstance(rho, list):
    #         self.beta = beta
    #         self.gamma = gamma
    #         self.rho = rho

    #         self.evolve()

    def __init__(self, i, S0, I0, R0, D0, N, t, beta, gamma, rho, Tr_matrix, total_cities, mainDf):
        self.D0 = D0
        self.rho = 0
        super().__init__( i, S0, I0, R0, N, t, beta, gamma, Tr_matrix, total_cities, mainDf)
        
        if all(isinstance(param, list) for param in [beta, gamma, rho]):
            for i in range(len(beta)):
                self.beta = beta[i]
                self.gamma = gamma[i]
                self.rho = rho[i]
                self.evolve()
        elif not all(isinstance(param, list) for param in [beta, gamma, rho]):
            self.beta = beta
            self.gamma = gamma
            self.rho = rho
            self.evolve()
        else:
            raise("Parameters beta, gama, and rho must have the same data type.")
    
    def SIRD_Model(self, y, t, N, beta, gamma, rho): 
        """
            Calculates all parameters of the SIRD Model.

            Args:
                y: initial parameters of the model. 
                t: time for the model to evolve. 
                N: total population of a city
                beta: the rate of infection.
                gamma: the rate of recovery.
                rho: the rate of mortality.

            Returns:
                dSdt: change in the number of susceptible individuals with time.
                dIdt: change in the number of infected individuals with time.
                dRdt: change in the number of removed (deceased or immune) individuals with time.
                dDdt: change of the deceased individuals with time.
        """
        S, I, R, D = y
        dSdt = -beta * S * (I+self.Tr(self.i)) / N
        dIdt = beta * S * (I+self.Tr(self.i)) / N - gamma * I
        dRdt = gamma * I -  rho * I
        dDdt = rho * I
        
        return dSdt, dIdt, dRdt, dDdt

    def evolve(self):
        """
			Computes the deriviative of the SIRD model at time t. 
			Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
			Initial conditions: S0, I0, R0, D0.
			Args:
				None.
			Returns:
				solution: an array-like containing the value of SIRD model for each desired time in t.
		"""
        self.solution = odeint(self.SIRD_Model, [self.S0, self.I0, self.R0, self.D0], 
                                self.t, args=(self.N, self.beta, self.gamma, self.rho))
        
        return self.solution
