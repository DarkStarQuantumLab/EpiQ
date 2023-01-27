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
import numpy as np
from typing import Tuple, Any


class EpidemiologyModel:
    """
        Parent class for SIR, SIRD, SEIRD, SEIR Epidemiological Models.
    """

    def __init__(self, i, S0, I0, R0, N, t, beta, gamma, Tr_matrix, total_cities, mainDf):
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.N = N
        self.t = t
        self.i = i
        self.Tr_matrix = Tr_matrix
        if mainDf is not None:
            self.total_cities = len(mainDf)
        else:
            self.total_cities = total_cities


    def Tr(self, city:int, lockdown=False):
        #TODO: finish comments
        """
            Transfer function that let transmission of virus between two or more cities.

            Args:
                city: (integer) indecis of the cities to optimize lockdown procedures.
                lockdown: (boolean), default value is False.
            Returns:
        """
        lockdown_init_list = np.ones(self.total_cities)
        sum_ = 0
        for city_index in range(self.total_cities):
            if city_index != city:
                if lockdown == False:
                    if lockdown_init_list[city_index] == 1:
                        sum_ += (self.S0 * self.Tr_matrix[city][city_index])
                    else:
                        sum_ += 0.25 * (self.S0 * self.Tr_matrix[city][city_index])
                    return sum_
        return 0 