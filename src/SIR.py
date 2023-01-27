from scipy.integrate import odeint
import numpy as np
from .base_epidimiology_model_class import EpidemiologyModel

class SIR(EpidemiologyModel):
	"""
	SIR (Susceptible Infected Recover) Class of Epidemiological Model.
	"""


	def __init__(self, i, S0, I0, R0, N, t, beta, gamma, Tr_matrix, total_cities, mainDf):
		super().__init__(i, S0, I0, R0, N, t, beta, gamma, Tr_matrix, total_cities, mainDf)

		if all(isinstance(param, list) for param in [beta, gamma]):
			for i in range(len(beta)):
				self.beta = beta[i]
				self.gamma = gamma[i]
				self.evolve()
		elif not all(isinstance(param, list) for param in [beta, gamma]):
			self.beta = beta
			self.gamma = gamma
			self.evolve()
		else:
			raise("Parameters beta and gama must have the same data type.")
 
	
	def SIR_model(self, y, t, N, beta, gamma):
		"""
            Calculates all parameters of the SIR Model.

            Args:
                y: initial parameters of the model. 
                t: time for the model to evolve. 
                N: total population of a city
                beta: the rate of infection.
                gamma: the rate of recovery.

            Returns:
                dSdt: change in the number of susceptible individuals with time.
                dIdt: change in the number of infected individuals with time.
                dRdt: change in the number of removed (deceased or immune) individuals with time.
        """
		S, I, R = y
		dS_dt = -beta * S * (I + self.Tr(self.i)) / N
		dI_dt = beta * S * (I + self.Tr(self.i)) / N - gamma * I
		dR_dt = gamma * I

		return [dS_dt, dI_dt, dR_dt]
	

	def evolve(self):
		"""
			Computes the deriviative of the SIR model at time t. 
			Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
			Initial conditions: S0, I0, R0.
			Args:
				None.
			Returns:
				solution: an array-like containing the value of SIR model for each desired time in t.
		"""
		self.solution=odeint(self.SIR_model, [self.S0, self.I0, self.R0], 
							self.t, args=(self.N, self.beta, self.gamma))

		return self.solution