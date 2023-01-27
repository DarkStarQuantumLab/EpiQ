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

from .epidemiology import *
import pandas as pd
import numpy as np
from copy import deepcopy
import dimod
from dimod import AdjVectorBQM
from dwave.system import DWaveSampler
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dwave.system.composites import EmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler
from typing import List, Dict
from math import log, ceil

class Optimization:
    
    def __init__(self, model_name:str, data_frame:list,
                distance_dataframe:list, max_infected_limit:int, model_property:dict, 
                lockdown_strength:int, solver_type:str,):
        self.solver_type = solver_type
        self.max_infected_limit = max_infected_limit
        self.opencity = []
        self.max_gdp = 0
        self.max_infected = 0
        self.model_name = model_name
        self.data_frame = data_frame
        self.fixed_dataset = data_frame
        self.city = self.data_frame['place'].to_list()
        self.total_cities = len(self.city)
        self.model_property = model_property
        self.distance_dataframe = distance_dataframe
        self.lockdown_strength = lockdown_strength
        self.infected = self.data_frame[self.model_property['min']].to_list()
        self.data_dict = data_frame.set_index('place').T.to_dict('list')
        self.output_sol = []
        self.lockdown_list = []
        self.global_count = 0 # for 1st time take all element of the list, then for rest exclude 1st as its repeat
        self.weight_selected = 'Exposed'
        
        count = 0
        
        if self.model_name == 'SEIRDV':
            self.model_name = 'SEIivIcvRDVIm'

        for x in self.data_dict:
            if self.model_name == 'SIR':
                self.data_dict[x] = {
                        'susceptible': [self.data_frame['population'][count]], 
                        'infected': [self.data_frame['infected'][count]],
                        'recovered': [self.data_frame['recovered'][count]]
                }
            elif self.model_name == 'SIRD':
                self.data_dict[x] = {
                        'susceptible': [self.data_frame['population'][count]], 
                        'infected': [self.data_frame['infected'][count]],
                        'recovered': [self.data_frame['recovered'][count]], 
                        'dead': [self.data_frame['dead'][count]]
                }
            elif self.model_name == 'SEIR':
                self.data_dict[x] = {
                        'susceptible': [self.data_frame['population'][count]],
                        'exposed':[self.data_frame['exposed'][count]], 
                        'infected': [self.data_frame['infected'][count]],
                        'recovered': [self.data_frame['recovered'][count]]
                }
            elif self.model_name == 'SEIRD':
                self.data_dict[x] = {
                        'susceptible': [self.data_frame['population'][count]], 
                        'exposed':[self.data_frame['exposed'][count]], 
                        'infected': [self.data_frame['infected'][count]],
                        'recovered': [self.data_frame['recovered'][count]], 
                        'dead':[self.data_frame['dead'][count]]
                }
            elif self.model_name == 'SEIivIcvRDVIm':
                self.data_dict[x] = {
                        'susceptible': [self.data_frame['population'][count]], 
                        'exposed': [self.data_frame['exposed'][count]], 
                        'infected_iv': [self.data_frame['infected_iv'][count]], 
                        'infected_cv': [self.data_frame['infected_cv'][count]], 
                        'recovered': [self.data_frame['recovered'][count]], 
                        'dead': [self.data_frame['dead'][count]], 
                        'vaccinated': [self.data_frame['vaccinated'][count]], 
                        'immunity': [self.data_frame['immunity'][count]]
                }
            count = count + 1

        self.data_process()

    def data_process(self):
        """
        Prepare the dataset.

        Args:
            None.
        Returns:
            None.
        """
        self.total_state=len(self.data_frame)
        states = list(self.distance_dataframe.columns)
        Distances = np.zeros((len(states), len(states)))
        
        for i in range(len(states)):
            for j in range(len(states)):
                if i > j:
                    Distances[j][i] = self.distance_dataframe[states[i]][states[j]]
                elif i < j:
                    Distances[j][i] = self.distance_dataframe[states[j]][states[i]]
                         
        self.Tr_matrix = np.zeros((self.total_state, self.total_state))
        
        for row in range(self.total_state):
            for column in range(self.total_state):
                distance = Distances[column][row]
                k = 2
                self.Tr_matrix[column][row] =  1 / ((distance / k) + 1)

    def optimize(self, time:list, interval:int, verbose:bool=False):
        """
        Args:
            time: (list)
            interval: (int)
            verbose: (bool), default is False.
        Returns:
            None.
        """
        self.interval = interval
        self.time = [x for x in range(0, time, interval)]
        self.time_points = [x for x in range(0, interval)]
        self.verbose = verbose

        print("=========================================\nTime: "+ str(time) + "Interval: " + str(interval))

        before_dataset = []
        for i in self.data_dict.values():
            temp_list = []
            for j in i.values():
                temp_list.append(j[-1])
            before_dataset.append(temp_list)
        before_dataset = pd.DataFrame(before_dataset,columns = list(list(self.data_dict.values())[0].keys()))
        print("\n\nDataSet Before: \n",before_dataset.to_string())

        self.run_lockdown(verbose)

        x = []
        for i in self.data_dict:
            temp = []
            for j in self.data_dict[i]:
                temp.append(self.data_dict[i][j])
            x.append(temp)
        x = np.array(x)
        self.output_sol = []
        for i in x:
            temp =  np.array(i)
            temp = temp.transpose()
            self.output_sol.append(temp)

        states_name = [city for city in self.data_frame.place]
        output_data = {'State Days -->': states_name}
        X_O_list = deepcopy(self.lockdown_list)
        days = 0
        
        for phase_t in range(len(self.lockdown_list)):
            for city in range(self.total_cities):
                if self.lockdown_list[phase_t][city] == 0:
                    X_O_list[phase_t][city] = 'O'
                else:
                    X_O_list[phase_t][city] = 'X'
            
        for t in range(len(self.lockdown_list)):
            output_data[days] = X_O_list[t]
            days += self.interval

        after_dataset = []
        for i in self.output_sol:
            after_dataset.append(i[-1].tolist())
            
        after_dataset = pd.DataFrame(after_dataset,columns = list(list(self.data_dict.values())[0].keys()))
        print("\n\nDataSet After: \n",after_dataset.to_string())

        df_marks = pd.DataFrame(output_data)
        
        print("\nO: Open City, X: Close city\n", df_marks)

    def run_lockdown(self, verbose):
        """
        Classify cities as open, closed through knapsack algorithms. The knapsack problem is solved
        either on quantum hardaware, or on classical computing device (CPU) 
        or as a simulated annealing algorithm.

        Args: 
            verbose
        """
        self.cost = self.data_frame[self.model_property['max']].to_list() #gdp
        self.weight=[0 for x in range(len(self.city))] #infected
        self.max_list = []
        open_city = []
        
        for ti in self.time:
            count = 0
            self.opencity = []
            lockdown_temp= [1 for i in range(self.total_cities)] # 0 means open

            if verbose:
                print('\n\n=======================================')
                print('at time', int(ti), 'days')
                print('-------------------------------------')
                print('Population from data_frame', sum(self.data_frame.population))
                print('Hospital capacity', self.max_infected_limit)
                print('---------------------------------------')
                print('city', 'cost=gdp', 'weight=' + self.weight_selected)
                print('---------------------------------------')
            
            for c in self.city:
                if 'E' in self.model_name or 'e' in self.model_name:
                    self.weight[count] = self.data_dict[c]['exposed'][-1:][0]
                
                else:
                    self.weight[count] = self.data_dict[c]['infected'][-1:][0]
                    self.weight_selected = 'Infected'
               
                count = count + 1

            for c in range(self.total_cities):
                 if verbose:
                    if 'E' in self.model_name or 'e' in self.model_name:
                        print(self.data_frame['place'][c],self.data_frame['GDP'][c],self.data_frame['exposed'][c])
                    else:
                        print(self.data_frame['place'][c],self.data_frame['GDP'][c],self.data_frame['infected'][c])

            if verbose:
                print('---------------------------------')
                print('total cost=Total GDP', sum(self.cost))
                print('total weight= Total ' + self.weight_selected, sum(self.weight))
                print('-------------------------------------')

            if self.solver_type == 'classical':
                self.max_gdp, self.max_infected, open_city = self.knapsack_classical(self.cost, self.weight)
                self.opencity = [open_city[x][3] for x in range(len(open_city))]
            elif self.solver_type == 'quantum':
                solution_temp = self.knapsack_quantum(list(self.data_frame['place']), self.cost, self.weight, self.verbose)
                (self.max_gdp, self.max_infected, open_cities, lockdown_list, closed_cities) = (solution_temp[0]['max_gdp'],
                    solution_temp[0]['max_infected'], solution_temp[0]['open_cities'], 
                    solution_temp[0]['lockdown_list'], solution_temp[0]['closed_cities'])
                for i in open_cities:
                    self.opencity.append(list(self.data_frame['place']).index(i))
            elif self.solver_type == 'simulated_annealing':
                solution_temp = self.knapsack_simulated(list(self.data_frame['place']), self.cost, self.weight, self.verbose)
                (self.max_gdp,self.max_infected, open_cities, lockdown_list, closed_cities) = (solution_temp[0]['max_gdp'],
                    solution_temp[0]['max_infected'], solution_temp[0]['open_cities'], 
                    solution_temp[0]['lockdown_list'], solution_temp[0]['closed_cities'])
                for i in open_cities:
                    self.opencity.append(list(self.data_frame['place']).index(i))
            else:
                raise Exception("Invalid solver type. Available solvers: classical, quantum, simulated_annealing.")

            if self.verbose:    
                self.show_data(self.opencity)

            self.max_list.append(self.max_gdp)
            
            for c in range(len(self.city)):
                ds = self.data_frame.iloc[[c]].copy(deep=True)
                for i in self.data_dict[self.city[c]]:
                    if i in ds.columns:
                       ds[i] = self.data_dict[self.city[c]][i][-1]
                if c in self.opencity:
                    model = Epidemiology(model_name=self.model_name, data_frame=ds,
                        distance_dataframe=self.distance_dataframe, time_points=self.time_points, 
                        show_dataset = verbose, lockdown=False, transfer_matrix = self.Tr_matrix, 
                        main_dataframe = self.fixed_dataset,lockdown_strength=self.lockdown_strength, 
                        solver_type=self.solver_type)

                    self.setData(model)
                    lockdown_temp[c] = 0
                else:
                    model = Epidemiology(model_name=self.model_name, data_frame=ds,
                        distance_dataframe=self.distance_dataframe, time_points=self.time_points,
                        show_dataset = verbose, lockdown=True, transfer_matrix = self.Tr_matrix, 
                        main_dataframe = self.fixed_dataset,lockdown_strength=self.lockdown_strength,
                        solver_type=self.solver_type)

                    self.setData(model)
            self.lockdown_list.append(lockdown_temp)
            self.global_count = self.global_count + 1

    def setData(self, model):
        """
        #TODO: add comments
        """
        if self.model_name == 'SIR':
            if self.global_count == 0:
                for i in range(len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['infected'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][2])
            else:
                for i in range(1,len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['infected'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][2])

        elif self.model_name == 'SIRD':
            if self.global_count == 0:
                for i in range(len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['infected'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][2])
                    self.data_dict[model.temp_dataset['City'][0]]['dead'].append(model.output_sol[0][i][3])
            else:
                for i in range(1,len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['infected'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][2])
                    self.data_dict[model.temp_dataset['City'][0]]['dead'].append(model.output_sol[0][i][3])
        elif self.model_name == 'SEIR':
            if self.global_count == 0:
                for i in range(len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['exposed'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['infected'].append(model.output_sol[0][i][2])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][3])
            else:
                for i in range(1, len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['exposed'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['infected'].append(model.output_sol[0][i][2])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][3])

        elif self.model_name == 'SEIRD':
            if self.global_count == 0:
                for i in range(len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['exposed'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['infected'].append(model.output_sol[0][i][2])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][3])
                    self.data_dict[model.temp_dataset['City'][0]]['dead'].append(model.output_sol[0][i][4])

            else:
                for i in range(1, len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['exposed'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['infected'].append(model.output_sol[0][i][2])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][3])
                    self.data_dict[model.temp_dataset['City'][0]]['dead'].append(model.output_sol[0][i][4])

        elif self.model_name == 'SEIivIcvRDVIm':

            if self.global_count == 0:
                for i in range(len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['exposed'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['infected_iv'].append(model.output_sol[0][i][2])
                    self.data_dict[model.temp_dataset['City'][0]]['infected_cv'].append(model.output_sol[0][i][3])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][4])
                    self.data_dict[model.temp_dataset['City'][0]]['dead'].append(model.output_sol[0][i][5])
                    self.data_dict[model.temp_dataset['City'][0]]['vaccinated'].append(model.output_sol[0][i][6])
                    self.data_dict[model.temp_dataset['City'][0]]['immunity'].append(model.output_sol[0][i][7])

            else:
                for i in range(1, len(model.output_sol[0])):
                    self.data_dict[model.temp_dataset['City'][0]]['susceptible'].append(model.output_sol[0][i][0])
                    self.data_dict[model.temp_dataset['City'][0]]['exposed'].append(model.output_sol[0][i][1])
                    self.data_dict[model.temp_dataset['City'][0]]['infected_iv'].append(model.output_sol[0][i][2])
                    self.data_dict[model.temp_dataset['City'][0]]['infected_cv'].append(model.output_sol[0][i][3])
                    self.data_dict[model.temp_dataset['City'][0]]['recovered'].append(model.output_sol[0][i][4])
                    self.data_dict[model.temp_dataset['City'][0]]['dead'].append(model.output_sol[0][i][5])
                    self.data_dict[model.temp_dataset['City'][0]]['vaccinated'].append(model.output_sol[0][i][6])
                    self.data_dict[model.temp_dataset['City'][0]]['immunity'].append(model.output_sol[0][i][7])
            

    def knapsack_classical(self, GDP:list, infected:list):
        """
        Classical implementation of the knapsack problem.

        Args:
            GPD: (list)
            infected: (list)

        Returns:
            max_gdp:
            max_infected:
            opencity:
        """
        res =- 1
        temp = []        
        K = [[0 for w in range(self.max_infected_limit + 1)]
                for i in range(self.total_cities + 1)]
        for i in range(self.total_cities + 1):
            for w in range(self.max_infected_limit + 1):
                if i == 0 or w == 0:
                    K[i][w] = 0
                elif infected[i - 1] <= w:  
                    K[i][w] = max(GDP[i - 1] + K[i - 1][w - int(infected[i - 1])],K[i - 1][w])
                else:
                    K[i][w] = K[i - 1][w]
        res = K[self.total_cities][self.max_infected_limit]
        self.max_gdp = res
        w = self.max_infected_limit
        
        for i in range(self.total_cities, 0, -1):
            if res <= 0:
                break
            if res == K[i - 1][int(w)]:
                continue
            else:
                temp.append([infected[i - 1],GDP[i-1]])
                res = res - GDP[i - 1]
                w = w - infected[i - 1]
        
        for i in temp:
            for j in range(self.total_cities):
                if( [ infected[j],GDP[j] ] == i  ):
                    self.max_infected=i[0]
                    self.opencity.append([self.city[j], i[0], i[1], j]) 
        
        temp_2= [0] * len(self.city)
        for i in self.opencity:
            temp_2[(i[-1])] = 1
        
        return self.max_gdp, self.max_infected, self.opencity

    def show_data(self,openCityList):
        #TODO: add comments
        """
        """
        print("\nModel Name:" + self.model_name)
        print("--------------------------------------------------------")
        print ('{:<25} {:<25} {:<25}'.format("city(No-Lockdown)",self.weight_selected,'GDP'))
        print("--------------------------------------------------------")
        for i in openCityList:
            print ("{:<25} {:<25} {:<25}".format(str(self.city[i]),str(self.data_dict[self.city[i]][self.weight_selected.lower()][-1]),str(self.data_frame.iloc[i]['GDP'])))
            # print(+"            | "++"            | "+))
        print("\n\nMaximum Limit For Infected(Given By User): " + str(self.max_infected_limit))
        print("Maximum GDP: " + str(self.max_gdp))
        print("Maximum " + self.weight_selected + ": " + str(self.max_infected))

    def knapsack_bqm(self, cities:list, values:list, weights:list, total_capacity:int):
        """
            Constact BQM (Binary Qundratic Model) for the knapsack problem.
            Args:
                cities: (list)
                values: (list)
                weights: (list)
                total_capacity:
            Return:
                bqm: instance of 'BinaryQuadraticModel'
        """
        bqm = AdjVectorBQM(dimod.Vartype.BINARY)
        lagrange = max(values)

        # Number of objects
        x_size = len(values)
        max_y_index = ceil(log(total_capacity))

        y = [2**n for n in range(max_y_index - 1)]
        y.append(total_capacity + 1 - 2**(max_y_index - 1))

        # Hamiltonian xi-xi terms
        for k in range(x_size):
            bqm.set_linear(
                cities[k],
                lagrange * (weights[k] ** 2) - values[k])

        # Hamiltonian xi-xj terms
        for i in range(x_size):
            for j in range(i + 1, x_size):
                key = (cities[i], cities[j])
                bqm.quadratic[key] = 2 * lagrange * weights[i] * weights[j]

        # Hamiltonian y-y terms
        for k in range(max_y_index):
            bqm.set_linear('y' + str(k), lagrange *
                           (y[k]**2) )

        # Hamiltonian yi-yj terms
        for i in range(max_y_index):
            for j in range(i + 1, max_y_index):
                key = ('y' + str(i), 'y' + str(j))
                bqm.quadratic[key] = 2 * lagrange * y[i] * y[j]

        # Hamiltonian x-y terms
        for i in range(x_size):
            for j in range(max_y_index):
                key = (cities[i], 'y' + str(j))
                bqm.quadratic[key] = -2 * lagrange * weights[i] * y[j]

        return bqm

    def _results_postprocessing(self, samplesets, cities:list, cost:list, weight:list) -> dict:
        """
        Postprocess the results.

        Args:
            samplesets: a 'dimod.SampleSet SamplesArray' solution set.
            cities: a list of cities to run lockdown procedure on.  
            cost: (list)
            weight: (list)
        Returns:
            solution_set: a dictionary with the lockdown recomendation.
        """
        df__ = pd.DataFrame({'city': cities, 'gdp': cost, 'sick': weight})
        df__ = df__.set_index('city')
       
        city_lockdown  = []
        gdp_temp = 0
        solution_set = []
        open_cities = []
        closed_cities = []

        for k, v in samplesets.first.sample.items():
            if k in cities:
                if v == 1:
                    open_cities.append(k)
                else:
                    closed_cities.append(k)

        # lockdown: 0, open 1
        for city in range(self.total_cities):
            if self.data_frame.place[city] in open_cities:
                city_lockdown.append(1)
            else:
                city_lockdown.append(0)
        gdp_temp = sum(df__.loc[open_cities]['gdp'])
        
        solution_set.append({
            'open_cities': open_cities,
            'closed_cities': closed_cities,
            'lockdown_list': city_lockdown,
            'energy': samplesets.first.energy,
            'max_gdp': sum(df__.loc[open_cities]['gdp']),
            'max_infected': int(round(sum(df__.loc[open_cities]['sick'])))
        })

        return solution_set

    def knapsack_quantum(self, cities:list, cost:list, weight:list, verbose=False):
        """
        Solves the knapsack problem on physical quantum hardware.
        Args:
            cities: (list)
            cost: (list)
            weight: (list)
            verbose: (bool) default is False.
        Return:
            solution_set: 'dimod.SampleSet SamplesArray' of a solution.
        """

        sampler = DWaveSampler()
        sampler = EmbeddingComposite(sampler)
        # this will fail for a larger problem (around 36 or more terms for Advantage)
        # For a prolem with larger size consider to pass quantum annealing parameters
        bqm = self.knapsack_bqm(cities, cost, weight, self.max_infected_limit)
        # number of samples was not provided. Number of samples has to be submitted based on the problem size
        #TODO: add option to submit used-defined hardware parameters (chain_strenght, anneal schedule, etc.)
        samplesets = sampler.sample(bqm)
        
        return self._results_postprocessing(samplesets, cities, cost, weight)

    def knapsack_simulated(self, cities:list, cost:list, weight:list, verbose=False):
        """
        Solves the knapsack problem using Dwave Simulated Annealing algorithm.
        Args:
            cities: (list)
            cost: (list)
            weight: (list)
            verbose: (bool) default is False.
        Return:
            solution_set: 'dimod.SampleSet SamplesArray' of a solution.
        """
        sampler = SimulatedAnnealingSampler()
        bqm = self.knapsack_bqm(cities, cost, weight, self.max_infected_limit)
        num_reads = 1000
        # an arbitrary choise of the num_reads parameter. its value depends on the data structure and problem type.
        if bqm.num_variables > 100:
            num_reads = 5000
        samplesets = sampler.sample(bqm, num_reads=num_reads)
        
        return self._results_postprocessing(samplesets, cities, cost, weight)