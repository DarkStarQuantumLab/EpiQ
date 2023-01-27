# Copyright DarkStarQuantumLab, Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from scipy.integrate import odeint
import numpy as np
from .base_epidimiology_model_class import EpidemiologyModel

class SEIR(EpidemiologyModel):
    """
    A class of Epidemiology model SEIR (Susceptible Exposed Infected Recovered).

    Model Description
    -----------------


    Parameters
    ----------
    S0 : int 
        Initial population in Susceptible Category
    E0 : int
        Initial population in Exposed Category
    I0 : int
        Initial population in Infected Category
    R0 : int
        Initial population in Recovered Category
    N : int
        Total Population
    t : list
        time points upon which differential equation will be evaluated
    beta : floating-point

    gamma : floating-point

    delta :floating-point


    Methods
    -------
    SEIR_Model(y)
    """

    def __init__(self, i, S0, E0, I0, R0, N, t, beta, gamma, delta, 
                Tr_matrix, total_cities, mainDf):
        super().__init__(i, S0, I0, R0, N, t, beta, gamma, Tr_matrix, total_cities, mainDf)
        self.E0 = E0
        self.beta = 0
        self.gamma = 0
        self.delta = 0
        
        # TODO: use all, not all cases are handled. 
        if all(isinstance(param, list) for param in [beta, gamma, delta]):
            for i in range(len(beta)):
                self.beta = beta[i]
                self.gamma = gamma[i]
                self.delta = delta[i]
                print(self.beta,self.gamma,self.delta)
                self.evolve()
        elif not all(isinstance(param, list) for param in [beta, gamma, delta]):
            self.beta = beta
            self.gamma = gamma
            self.delta = delta
            self.evolve()
        else:
            raise("Parameters beta, gama, delta must have the same data type.")

    def SEIR_Model(self, y, t, N, beta, gamma, delta):
        """
            Calculates all parameters of the SIRD Model.

            Args:
                y: initial parameters of the model. 
                t: time for the model to evolve. 
                N: total population of a city
                beta: the rate of infection.
                gamma: the rate of recovery.
                delta: rate of progression from exposed to infectious
                    (the reciprocal is the incubation period).

            Returns:
                dSdt: change in the number of susceptible individuals with time.
                dIdt: change in the number of infected individuals with time.
                dRdt: change in the number of removed (deceased or immune) individuals with time.
                dEdt: change of exposed to desease individuals with time.
        """
        S, E, I, R = y
        dSdt = -beta* S * (I+self.Tr(self.i)) / N
        dEdt = beta * S * (I+self.Tr(self.i)) / N - delta * E
        dIdt = delta * E - gamma * I
        dRdt = gamma * I
    
        return [dSdt, dEdt, dIdt, dRdt]
    
    def evolve(self):
        """
			Computes the deriviative of the SEIR model at time t. 
			Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
			Initial conditions: S0, E0, I0, R0.
			Args:
				None.
			Returns:
				solution: an array-like containing the value of SEIR model for each desired time in t.
		"""
        self.solution=odeint(self.SEIR_Model, [self.S0, self.E0, self.I0, self.R0], 
                            self.t, args=(self.N, self.beta, self.gamma, self.delta))

        return self.solution