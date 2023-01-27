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