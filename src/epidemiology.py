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
import pandas as pd
from SIR import *
from SIRD import *
from SEIR import *
from SEIRD import *
from SEIivIcvRDVIm import *
import numpy as np
pd.options.display.float_format = '{:20,.2f}'.format

class Epidemiology:

	"""Epidemiology Class to evolve infection over compartments
	Supported Epidemiology Models

		SIR (Susceptible Infected Recovered) Model
		SIRD (Susceptible Infected Recovered Dead) Model
		SEIR (Susceptible Exposed Infected Recovered) Model
		SEIRD (Susceptible Exposed Infected Recovered) Model
		SEIRDV :
			S: Susceptible, E: Exposed, I_iv: Infected Incomplete Vaccination, 
			I_cv: Infected Complete Vaccination, R: Recovered, D: Dead, I_m: Immunity

	...

	Attributes 
	----------

	(User Defined)

	model_name: string
				The Name of the Model
	solver_type: (string) 'classical', 'quantum', or 'simulated_annealing'. 
				'classical' provides standard algorithm of solving
				'quantum' option executes the optimization on DWave's quantum hardware
				'simulated_annealing' executes the optimization part as classical simulation 
				of the quantum hardware.
	time_points: List 
			     list of time points
	data_frame: pandas DataFrame
				The initial dataset (csv format)
	distance_dataframe: pandas DataFrame
						Distance dataset, to be used in Transfer Function
						Optional, for SEIRDV Model
	show_dataset: bool, default: True
			   	  Output the final dataset after evolution of models

	...

	In-Methods Parameters
	---------------------

	lockdown: bool, default: False
	transfer_matrix: list, default:
	main_dataframe: list, defalut:
	lockdown_strength: Integer/Float, default:
	total_cities: int
	output_sol: list
	temp_dataset: 
	global_infected: list

	Methods

	data_process()
		Process the initial data for evolution of the Models
	process()
		Calls the necessary packages to do evolution of the Models
	"""
	
	def __init__(self, model_name:str, time_points:list, data_frame:list, solver_type:str,
				distance_dataframe:list = None, show_dataset:bool = True, 
				lockdown:bool =False, transfer_matrix:list = None, 
				main_dataframe: list = None, lockdown_strength: float =None):
		#seems that the solver type is used in model plot only.
		self.solver_type = solver_type 
		self.model_name = model_name
		self.time_points = time_points
		self.data_frame = data_frame
		self.distance_dataframe = distance_dataframe
		self.show_dataset = show_dataset
		
		self.lockdown = lockdown
		self.transfer_matrix = transfer_matrix
		self.main_dataframe = main_dataframe
		self.lockdown_strength = lockdown_strength 
		
		self.total_cities = len(self.data_frame)
		self.output_sol = []
		self.temp_dataset = None
		self.global_infected = []
		
		self.data_process()
		self.process()

	def data_process(self):
		"""Does initial data processing: initialing distance matrix,
			initiating datasets, to be used in process() function.

			If the distance_dataframe is available for Epidemiology models
			like SIR, SEIR, SEIRD, SIRD, we make the distance matrix from
			the distance dataframe.

			From the Distance matrix we formulate the transfer_matrix, which
			will be the basis of the transmission of virus between cities.
			transfer_matrix will determine the number of people to be 
			transfer to and from the city.
		"""
		# Finding the total number of states from the given dataframe
		self.total_state = len(self.data_frame)

		if self.distance_dataframe is not None:
			states = list(self.distance_dataframe.columns)
			
			if self.transfer_matrix is  None:
				Distances = np.zeros((len(states), len(states)))
				for i in range(len(states)):
				    for j in range(len(states)):
				        if i > j:
				            Distances[j][i] = self.distance_dataframe[states[i]][states[j]]
				        elif i < j:
				            Distances[j][i] = self.distance_dataframe[states[j]][states[i]]

				self.transfer_matrix = np.zeros((self.total_state, self.total_state))
				
				for row in range(self.total_state):
				    for column in range(self.total_state):
				        distance = Distances[column][row]
				        k = 2
				        self.transfer_matrix[column][row] =  1 / ((distance / k) + 1)
		
		# Initilizaing data_frame dict which will store all the data duing evolution 
		# of Epidemiology model.
		if self.model_name == 'SEIRDV': self.model_name = 'SEIivIcvRDVIm'
		if self.model_name == 'SIR':
			try:
				self.data_frame['infected'].fillna(self.data_frame['infected'].mean(),inplace=True)
				self.data_frame['recovered'].fillna(self.data_frame['recovered'].mean(),inplace=True)
				self.data_frame['beta'].fillna(self.data_frame['beta'].mean(),inplace=True)
				self.data_frame['gamma'].fillna(self.data_frame['gamma'].mean(),inplace=True)			
			except:
				pass
			
		elif self.model_name == 'SIRD':
			try:
				self.data_frame['population'].fillna(self.data_frame['population'].mean(),inplace=True)
				self.data_frame['infected'].fillna(self.data_frame['infected'].mean(),inplace=True)
				self.data_frame['recovered'].fillna(self.data_frame['recovered'].mean(),inplace=True)
				self.data_frame['dead'].fillna(self.data_frame['dead'].mean(),inplace=True)
				self.data_frame['beta'].fillna(self.data_frame['beta'].mean(),inplace=True)
				self.data_frame['gamma'].fillna(self.data_frame['gamma'].mean(),inplace=True)
			except:
				pass

		elif self.model_name == 'SEIR' :	
			try:
				self.data_frame['population'].fillna(self.data_frame['population'].mean(),inplace=True)
				self.data_frame['exposed'].fillna(self.data_frame['exposed'].mean(),inplace=True)
				self.data_frame['infected'].fillna(self.data_frame['infected'].mean(),inplace=True)
				self.data_frame['recovered'].fillna(self.data_frame['recovered'].mean(),inplace=True)
				self.data_frame['beta'].fillna(self.data_frame['beta'].mean(),inplace=True)
				self.data_frame['delta'].fillna(self.data_frame['delta'].mean(),inplace=True)
				self.data_frame['gamma'].fillna(self.data_frame['gamma'].mean(),inplace=True)
			except:
				pass

		elif self.model_name == 'SEIRD':	
			try:
				self.data_frame['population'].fillna(self.data_frame['population'].mean(),inplace=True)
				self.data_frame['exposed'].fillna(self.data_frame['exposed'].mean(),inplace=True)
				self.data_frame['infected'].fillna(self.data_frame['infected'].mean(),inplace=True)
				self.data_frame['recovered'].fillna(self.data_frame['recovered'].mean(),inplace=True)
				self.data_frame['dead'].fillna(self.data_frame['dead'].mean(),inplace=True)
				self.data_frame['beta'].fillna(self.data_frame['beta'].mean(),inplace=True)
				self.data_frame['delta'].fillna(self.data_frame['delta'].mean(),inplace=True)
				self.data_frame['gamma'].fillna(self.data_frame['gamma'].mean(),inplace=True)
				self.data_frame['alpha'].fillna(self.data_frame['alpha'].mean(),inplace=True)
				self.data_frame['rho'].fillna(self.data_frame['rho'].mean(),inplace=True)
			except:
				pass

		elif(self.model_name == 'SEIivIcvRDVIm'):	
			try:
				self.data_frame['population'].fillna(self.data_frame['population'].mean(),inplace=True)
				self.data_frame['exposed'].fillna(self.data_frame['exposed'].mean(),inplace=True)
				self.data_frame['infected_iv'].fillna(self.data_frame['infected_iv'].mean(),inplace=True)
				self.data_frame['infected_cv'].fillna(self.data_frame['infected_cv'].mean(),inplace=True)
				self.data_frame['immunity'].fillna(self.data_frame['immunity'].mean(),inplace=True)
				self.data_frame['recovered'].fillna(self.data_frame['recovered'].mean(),inplace=True)
				self.data_frame['vaccinated'].fillna(self.data_frame['vaccinated'].mean(),inplace=True)
				self.data_frame['dead'].fillna(self.data_frame['dead'].mean(),inplace=True)
				self.data_frame['beta'].fillna(self.data_frame['beta'].mean(),inplace=True)
				self.data_frame['delta1'].fillna(self.data_frame['delta1'].mean(),inplace=True)
				self.data_frame['delta'].fillna(self.data_frame['delta'].mean(),inplace=True)
				self.data_frame['gamma_iv'].fillna(self.data_frame['gamma_iv'].mean(),inplace=True)
				self.data_frame['gamma_cv'].fillna(self.data_frame['gamma_cv'].mean(),inplace=True)
				self.data_frame['alpha'].fillna(self.data_frame['alpha'].mean(),inplace=True)
				self.data_frame['rho_iv'].fillna(self.data_frame['rho_iv'].mean(),inplace=True)
				self.data_frame['rho_cv'].fillna(self.data_frame['rho_cv'].mean(),inplace=True)
				self.data_frame['sigma'].fillna(self.data_frame['sigma'].mean(),inplace=True)
				self.data_frame['eta'].fillna(self.data_frame['eta'].mean(),inplace=True)
			except:
				pass

	def process(self):
		if self.model_name 	== 'SIR':
			self.model_SIR()
		elif self.model_name == 'SIRD':
			self.model_SIRD()			
		elif self.model_name == 'SEIR':
			self.model_SEIR()
		elif self.model_name == 'SEIRD' :
			self.model_SEIRD()
		elif self.model_name == 'SEIivIcvRDVIm':
			self.model_SEIRDV()
		else:
			raise Exception("Error: Check The Model Name.")
	
	def model_SIR(self):
		S0, I0, R0, N, name, beta, gamma, data = ([] for i in range(8))

		try:
			name += list(self.data_frame['place'])
			N += list(self.data_frame['population'])
			I0 += list(self.data_frame['infected'])
			R0 += list(self.data_frame['recovered'])
			beta += list(self.data_frame['beta'])
			gamma += list(self.data_frame['gamma'])
			
			[S0.append(N[i] - (I0[i] + R0[i])) for i in range(len(N))]

		except:
			raise Exception("Error: Check Your DataFrame.")

		for city_index in range(len(N)):

			param_beta = beta[city_index] - beta[city_index] * (self.lockdown_strength / 100) if self.lockdown else beta[city_index]
			
			sir_object = SIR(city_index ,S0[city_index], I0[city_index], R0[city_index], N[city_index], 
							self.time_points, param_beta, gamma[city_index], self.transfer_matrix, 
							self.total_state, self.main_dataframe)
			
			solution = sir_object.evolve()
			self.output_sol.append(solution)
		
		for index in range(len(solution)):
			self.global_infected.append(solution[index][1])

		for city_index in range(len(N)):
		    
		    data.append([name[city_index], N[city_index], self.output_sol[city_index][-1,0], 
		    			self.output_sol[city_index][-1,1],self.output_sol[city_index][-1,2], 
		    			sum([self.output_sol[city_index][-1,0], self.output_sol[city_index][-1,1], 
		    				self.output_sol[city_index][-1,2]])])
		
		self.temp_dataset = pd.DataFrame(data)
		self.temp_dataset.columns = ['City','Population','Suceptible','Infected','Recovered','Total*']

		if self.show_dataset:
			temp_first_df = pd.DataFrame(np.array([name,N,S0,I0,R0]).transpose())
			temp_first_df.columns =  ['City','Population','Suceptible','Infected','Recovered']
			
			print("Dataset before SIR evolution\n")
			print(temp_first_df.to_string())
			print("\nDataset after SIR evolution\n")
			print(self.temp_dataset.to_string())
			print("* Total: S + I + R\n\n")
		
	def model_SIRD(self):
		S0, I0, R0, D0, N, name, beta, gamma, rho, data = ([] for i in range(10))

		try:
			name += list(self.data_frame['place'])
			N += list(self.data_frame['population'])
			I0 += list(self.data_frame['infected'])
			R0 += list(self.data_frame['recovered'])
			D0 += list(self.data_frame['dead'])
			beta += list(self.data_frame['beta'])
			gamma += list(self.data_frame['gamma'])
			rho += list(self.data_frame['rho'])
			
			[S0.append(N[i] - (I0[i] + R0[i] + D0[i])) for i in range(len(N))]
			
		except:
			raise("Error: Check Your DataFrame.")

		for city_index in range(len(N)):
			param_beta = beta[city_index] - beta[city_index] * (self.lockdown_strength / 100) if self.lockdown else beta[city_index]
			
			sird_object = SIRD(city_index, S0[city_index], I0[city_index], R0[city_index], D0[city_index], N[city_index], 
								self.time_points, param_beta, gamma[city_index], rho[city_index], self.transfer_matrix, 
								self.total_state, self.main_dataframe)
			
			solution = sird_object.evolve()
			self.output_sol.append(solution)

		for city_index in range(len(N)):
		    
		    data.append([name[city_index], N[city_index], self.output_sol[city_index][-1,0], self.output_sol[city_index][-1,1],
		    			 self.output_sol[city_index][-1,2], self.output_sol[city_index][-1,3],sum([self.output_sol[city_index][-1,0], 
		    			 self.output_sol[city_index][-1,1], self.output_sol[city_index][-1,2],self.output_sol[city_index][-1,3]])])
		
		self.temp_dataset = pd.DataFrame(data)
		self.temp_dataset.columns = ['City','Population','Suceptible','Infected','Recovered','Dead','Total*']
		
		
		if self.show_dataset:
			temp_first_df = pd.DataFrame(np.array([name,N,S0,I0,R0,D0]).transpose())
			temp_first_df.columns =  ['City','Population','Suceptible','Infected','Recovered','Dead']
			
			print("Dataset before SIRD evolution\n")
			print(temp_first_df.to_string())
			print("Dataset after SIRD evolution\n")
			print(self.temp_dataset.to_string())
			print("* Total: S + I + R + D\n\n")

	def model_SEIR(self):
		S0, E0, I0, R0, N, name, beta, delta, gamma, data = ([] for i in range(10))

		try:
			name += list(self.data_frame['place'])
			N += list(self.data_frame['population'])
			E0 += list(self.data_frame['exposed'])
			I0 += list(self.data_frame['infected'])
			R0 += list(self.data_frame['recovered'])
			beta += list(self.data_frame['beta'])
			delta += list(self.data_frame['delta'])
			gamma += list(self.data_frame['gamma'])

			[S0.append(N[i] - (E0[i] + I0[i] + R0[i])) for i in range(len(N))]

		except:
			raise("Error: Check Your DataFrame")

		for city_index in range(len(N)):
			param_beta = beta[city_index] - beta[city_index] * (self.lockdown_strength / 100) if self.lockdown else beta[city_index]
			
			seir_object = SEIR(city_index,S0[city_index], E0[city_index], I0[city_index], R0[city_index], N[city_index], self.time_points, 
							param_beta, delta[city_index], gamma[city_index],self.transfer_matrix,self.total_state,self.main_dataframe)
			
			solution = seir_object.evolve()
			self.output_sol.append(solution)

		for city_index in range(len(N)):
		    
		    data.append([name[city_index],N[city_index],self.output_sol[city_index][-1,0],self.output_sol[city_index][-1,1], 
		    			self.output_sol[city_index][-1,2], self.output_sol[city_index][-1,3],sum([self.output_sol[city_index][-1,0],
		    			self.output_sol[city_index][-1,1],self.output_sol[city_index][-1,2], self.output_sol[city_index][-1,3]])])
		
		self.temp_dataset = pd.DataFrame(data)
		self.temp_dataset.columns = ['City','Population','Suceptible','Exposed','Infected','Recovered','Total*']

		if self.show_dataset:
			temp_first_df = pd.DataFrame(np.array([name,N,S0,E0,I0,R0]).transpose())
			temp_first_df.columns =  ['City','Population','Suceptible','Exposed','Infected','Recovered']

			print("Dataset before SEIR evolution\n")
			print(temp_first_df.to_string())
			print("Dataset after SEIR evolution\n")
			print(self.temp_dataset.to_string())
			print("* Total: S + E + I + R\n\n")

	def model_SEIRD(self):
		S0, E0, I0, R0, D0, N, name, beta, delta, alpha, rho, gamma, data = ([] for i in range(13))

		try:
			name += list(self.data_frame['place'])
			N += list(self.data_frame['population'])
			E0 += list(self.data_frame['exposed'])
			I0 += list(self.data_frame['infected'])
			R0 += list(self.data_frame['recovered'])
			D0 += list(self.data_frame['dead'])
			beta += list(self.data_frame['beta'])
			delta += list(self.data_frame['delta'])
			alpha += list(self.data_frame['alpha'])
			rho += list(self.data_frame['rho'])
			gamma += list(self.data_frame['gamma'])

			[S0.append(N[i] - (E0[i] + I0[i] + R0[i] + D0[i])) for i in range(len(N))]

		except:
			raise("Error: Check Your DataFrame")

		for city_index in range(len(N)):
			param_beta = beta[city_index] - beta[city_index] * (self.lockdown_strength / 100) if self.lockdown else beta[city_index]

			serid_object = SEIRD(city_index,S0[city_index], E0[city_index], I0[city_index], R0[city_index], D0[city_index], N[city_index], 
								self.time_points, param_beta, delta[city_index], alpha[city_index], rho[city_index], gamma[city_index],
								self.transfer_matrix,self.total_state, self.main_dataframe)
			
			solution = serid_object.evolve()
			self.output_sol.append(solution)
		
		for city_index in range(len(N)):
		    
		    data.append([name[city_index],N[city_index],self.output_sol[city_index][-1,0],self.output_sol[city_index][-1,1],
		    			self.output_sol[city_index][-1,2], self.output_sol[city_index][-1,3],self.output_sol[city_index][-1,4],
		    			sum([self.output_sol[city_index][-1,0],self.output_sol[city_index][-1,1],self.output_sol[city_index][-1,2],
		    			self.output_sol[city_index][-1,3],self.output_sol[city_index][-1,4]])])

		self.temp_dataset = pd.DataFrame(data)
		self.temp_dataset.columns = ['City','Population','Susceptible','Exposed','Infected','Recovered','Dead','Total*']

		if self.show_dataset:
			temp_first_df = pd.DataFrame(np.array([name, N, S0, E0, I0, R0, D0]).transpose())
			temp_first_df.columns =  ['City','Population','Susceptible','Exposed','Infected','Recovered','Dead']

			print("Dataset before SEIRD evolution\n")
			print(temp_first_df.to_string())
			print("Dataset after SEIRD evolution\n")
			print(self.temp_dataset.to_string())
			print("* Total: S + E + I + R + D\n\n")
	
	def model_SEIRDV(self):
		S0, E0, I0_iv, I0_cv, R0, D0, V0, N, Im0, name, beta, delta, delta1, alpha, \
		eta, gamma_iv, gamma_cv, sigma, rho_iv, rho_cv, data = ([] for i in range(21))

		try:
			name += list(self.data_frame['place'])
			N += list(self.data_frame['population'])
			E0 += list(self.data_frame['exposed'])
			I0_cv += list(self.data_frame['infected_cv'])
			I0_iv += list(self.data_frame['infected_iv'])
			Im0 += list(self.data_frame['immunity'])
			R0 += list(self.data_frame['recovered'])
			D0 += list(self.data_frame['dead'])
			V0 += list(self.data_frame['vaccinated'])
			beta += list(self.data_frame['beta'])
			delta1 += list(self.data_frame['delta1'])
			delta += list(self.data_frame['delta'])
			alpha += list(self.data_frame['alpha'])
			eta += list(self.data_frame['eta'])
			gamma_iv += list(self.data_frame['gamma_iv'])
			gamma_cv += list(self.data_frame['gamma_cv'])
			rho_iv += list(self.data_frame['rho_iv'])
			rho_cv += list(self.data_frame['rho_cv'])
			sigma += list(self.data_frame['sigma'])

			[S0.append(N[i] - (E0[i] + I0_cv[i] + I0_iv[i] + Im0[i] + R0[i] + D0[i] + V0[i])) for i in range(len(N))]

		except:
			raise("Error: Check Your DataFrame")

		for city_index in range(len(N)):
			param_beta = beta[city_index] - beta[city_index] * (self.lockdown_strength / 100) if self.lockdown else beta[city_index]

			seirdv_object = SEIivIcvRDVIm(S0[city_index], E0[city_index], I0_iv[city_index], I0_cv[city_index], R0[city_index], 
										D0[city_index], V0[city_index], Im0[city_index],N[city_index], self.time_points, param_beta, 
										delta[city_index], delta1[city_index], alpha[city_index], eta[city_index], gamma_iv[city_index], 
										gamma_cv[city_index], rho_iv[city_index], rho_cv[city_index], sigma[city_index])
			
			solution = seirdv_object.evolve()
			self.output_sol.append(solution)
		
		for city_index in range(len(N)):
			
		    data.append([name[city_index],N[city_index], self.output_sol[city_index][-1,0], self.output_sol[city_index][-1,1], 
		    		self.output_sol[city_index][-1,2], self.output_sol[city_index][-1,3], self.output_sol[city_index][-1,4], 
		    		self.output_sol[city_index][-1,5], self.output_sol[city_index][-1,6], self.output_sol[city_index][-1,7], 
		    		sum([self.output_sol[city_index][-1,0], self.output_sol[city_index][-1,1], self.output_sol[city_index][-1,2],
		    		self.output_sol[city_index][-1,3], self.output_sol[city_index][-1,4], self.output_sol[city_index][-1,5], 
		    		self.output_sol[city_index][-1,6], self.output_sol[city_index][-1,7]])])
		
		self.temp_dataset = pd.DataFrame(data)
		self.temp_dataset.columns = ['City','Population','Susceptible','Exposed','Infected_incomplete_vaccination', 
									'Infected_complete_vaccination','Recovered','Dead', 'Vaccinated', 'Immunity', 'Total*']
		if self.show_dataset:
			temp_first_df = pd.DataFrame(np.array([name, N, S0, E0, I0_iv, I0_cv, R0, D0, V0, Im0]).transpose())
			temp_first_df.columns =  ['City','Population','Susceptible','Exposed','Infected_incomplete_vaccination', 
									'Infected_complete_vaccination','Recovered','Dead', 'Vaccinated', 'Immunity']
			print("Dataset before SEIRD evolution\n")
			print(temp_first_df.to_string())
			print("Dataset after SEIivIcvRDVIm evolution\n")
			print(self.temp_dataset.to_string())
			print("* Total: S + E + I_iv + I_cv + R + D + V + Im\n\n")	