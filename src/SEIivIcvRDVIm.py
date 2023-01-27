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

class SEIivIcvRDVIm:
    def __init__(self, S0 , E0, I0_iv, I0_cv, R0, D0, V0, Im0, N, t, beta, delta, 
                    delta1, alpha, eta, gamma_iv, gamma_cv, rho_iv, rho_cv, sigma):
        self.S0 = S0
        self.E0 = E0
        self.I0_iv = I0_iv
        self.I0_cv = I0_cv
        self.Im0 = Im0
        self.R0 = R0
        self.D0 = D0
        self.V0 = V0
        self.N = N
        self.t = t

        if all(isinstance(param, list) for param in [beta, delta, delta1, alpha, eta, gamma_iv, 
                                                    gamma_cv, rho_iv, rho_cv, sigma]):
            for i in range(len(beta)):
                self.beta = beta[i]
                self.gamma_iv = gamma_iv[i]
                self.gamma_cv = gamma_cv[i]
                self.delta1 = delta1[i]
                self.delta = delta[i]
                self.alpha = alpha[i]
                self.eta = eta[i]
                self.rho_iv = rho_iv[i]
                self.rho_cv = rho_cv[i]
                self.sigma = sigma[i]
                self.evolve()
        elif not all(isinstance(param, list) for param in [beta, delta, delta1, alpha, eta, gamma_iv, 
                                                    gamma_cv, rho_iv, rho_cv, sigma ]):
            self.beta = beta
            self.gamma_iv = gamma_iv
            self.gamma_cv = gamma_cv
            self.delta1 = delta1
            self.delta = delta
            self.alpha = alpha
            self.eta = eta
            self.rho_iv = rho_iv
            self.rho_cv = rho_cv
            self.sigma = sigma
            self.evolve()
        else:
            raise("Parameters beta, delta, delta1, alpha, eta, gamma_iv, gamma_cv, rho_iv, rho_cv, sigma must have the same data type.")

    def SEIRDV_model(self, y,t, N, beta, sigma, delta, delta1, eta, alpha, 
                    gamma_iv, gamma_cv, rho_iv, rho_cv):
        """
            Calculates all parameters of the SEIRD Model.

            Args:
                y: initial parameters of the model. 
                t: time for the model to evolve. 
                N: total population of a city
                beta: the rate of infection.
                sigma:
                delta: rate of progression from exposed to infectious
                    (the reciprocal is the incubation period).
                delta1:  the incubation rate for the transition from Exposed to Infected states.
                eta:
				alpha: infection fatality rate.
                gamma_iv:
                gamma_cv:
				rho_iv: 
                rho_cv:

            Returns:
                dSdt: change in the number of susceptible individuals with time.
				dEdt: dEdt: change of exposed to desease individuals with time.
                dI_ivdt: removal rate relates to the average infectious period
                dI_cvdt:
                dRdt: change in the number of removed (deceased or immune) individuals with time.
                dDdt: change of the deceased individuals with time.
                dVdt: change in the vaccinated population with time.
                dImdt: 
        """
        S, E, I_iv, I_cv, R, D, V, Im = y
        
        dSdt = -beta * S * (I_iv + I_cv) / N - sigma*S
        dEdt = beta * S * (I_iv + I_cv) / N - delta  * E - delta1 * E
        dI_ivdt = delta * E - (1 - alpha) * gamma_iv * I_iv - alpha * rho_iv * I_iv
        dI_cvdt = delta1 * E - (1 - alpha) * gamma_cv * I_cv - alpha * rho_cv * I_cv + (1 - eta) * V / N
        dRdt = (1 - alpha) * gamma_cv * I_cv + (1 - alpha) * gamma_iv * I_iv
        dDdt = alpha * rho_cv * I_cv + alpha * rho_iv * I_iv
        dVdt = sigma * S - V / N
        dImdt = eta * V / N
       
        return dSdt, dEdt, dI_ivdt, dI_cvdt, dRdt, dDdt, dVdt, dImdt

    def evolve(self):
        """
			Computes the deriviative of the SEIivIcvRDVIm model at time t. 
			Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
			Initial conditions: S0, E0, I0_iv, I0_cv, R0, D0, V0, Im0.
			Args:
				None.
			Returns:
				solution: an array-like containing the value of SEIivIcvRDVIm model for each desired time in t.
		"""
        self.solution = odeint(self.SEIRDV_model, [self.S0, self.E0, self.I0_iv, self.I0_cv, 
                                self.R0, self.D0, self.V0, self.Im0], self.t,
                                args=(self.N, self.beta, self.sigma,  self.delta, self.delta1,  
                                self.eta, self.alpha,  self.gamma_iv, self.gamma_cv, self.rho_iv, self.rho_cv ))
        
        return self.solution
