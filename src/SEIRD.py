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

class SEIRD(EpidemiologyModel):

	def __init__(self, i, S0, E0, I0, R0, D0, N, t, beta, delta, alpha, rho, gamma,
				Tr_matrix, total_cities, mainDf):
		super().__init__(i, S0, I0, R0, N, t, beta, gamma, Tr_matrix, total_cities, mainDf) 
		self.E0 = E0
		self.D0 = D0

		# TODO: use if all
		if all(isinstance(param, list) for param in [beta, gamma, delta, alpha, rho]):
			for i in range(len(beta)):
				self.beta = beta[i]
				self.gamma = gamma[i]
				self.delta = delta[i]
				self.alpha = alpha[i]
				self.rho = rho[i]
				self.evolve()
		elif not all(isinstance(param, list) for param in [beta, gamma, delta, alpha, rho]):
			self.beta = beta
			self.gamma = gamma
			self.delta = delta
			self.alpha = alpha
			self.rho = rho
			self.evolve()
		else:
			raise("Parameters beta, gama delta, alpha, and rho must have the same data type.") 

	def SEIRD_model(self, y, t, N, beta, gamma, delta, alpha, rho):
		"""
            Calculates all parameters of the SEIRD Model.

            Args:
                y: initial parameters of the model. 
                t: time for the model to evolve. 
                N: total population of a city
                beta: the rate of infection.
                gamma: the rate of recovery.
                delta: rate of progression from exposed to infectious
                    (the reciprocal is the incubation period).
				alpha: infection fatality rate.
				rho: epidemiological parameter.

            Returns:
                dSdt: change in the number of susceptible individuals with time.
				dEdt: dEdt: change of exposed to desease individuals with time.
                dIdt: change in the number of infected individuals with time.
                dRdt: change in the number of removed (deceased or immune) individuals with time.
                dDdt: change of the deceased individuals with time.
		"""
		S, E, I, R, D = y
		
		dS_dt = -beta * S * (I+self.Tr(self.i)) / N
		dE_dt = beta * S * (I+self.Tr(self.i)) / N - delta * E
		dI_dt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
		dR_dt = (1 - alpha) * gamma * I
		dD_dt = alpha * rho * I
		
		return [dS_dt, dE_dt, dI_dt, dR_dt, dD_dt]
	
	def evolve(self):
		"""
			Computes the deriviative of the SEIRD model at time t. 
			Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
			Initial conditions: S0, E0, I0, R0, D0.
			Args:
				None.
			Returns:
				solution: an array-like containing the value of SEIRD model for each desired time in t.
		"""
		self.solution = odeint(self.SEIRD_model, [self.S0, self.E0, self.I0, self.R0, self.D0], 
			self.t, args=(self.N, self.beta, self.gamma, self.delta, self.alpha, self.rho))
		
		return self.solution