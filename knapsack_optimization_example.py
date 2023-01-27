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

from src.optimization import *
from src.modelplot import *

if __name__ == "__main__":
    # load data
    dataFrame=pd.read_csv('data/data.csv') 
    timePoints=[x for x in range(1,50)]	
    distanceDataFrame = pd.read_csv('data/distance.csv', index_col = 0, header =0)

    model_property = {'max':'GDP', 'min':'infected', 'maxWeight':'Bed'}
    optimize_SIR_model = Optimization(model_name = 'SIR', data_frame=dataFrame, distance_dataframe=distanceDataFrame,
                                        solver_type='simulated_annealing', max_infected_limit=50050,
                                        model_property=model_property, lockdown_strength=80)
    optimize_SEIR_model = Optimization(model_name = 'SEIR', data_frame=dataFrame, distance_dataframe=distanceDataFrame,
                                        solver_type='simulated_annealing', max_infected_limit=50050,
                                        model_property=model_property, lockdown_strength=80)

    optimize_SIRD_model = Optimization(model_name = 'SIRD', data_frame=dataFrame, distance_dataframe=distanceDataFrame,
                                        solver_type='simulated_annealing', max_infected_limit=50050,
                                        model_property=model_property, lockdown_strength=80)
    optimize_SEIRD_model = Optimization(model_name = 'SEIRD', data_frame=dataFrame, distance_dataframe=distanceDataFrame,
                                        solver_type='simulated_annealing', max_infected_limit=50050,
                                        model_property=model_property, lockdown_strength=80)
    optimize_SEIivIcvRDVIm_model = Optimization(model_name = 'SEIivIcvRDVIm', data_frame=dataFrame, distance_dataframe=distanceDataFrame,
                                        solver_type='simulated_annealing', max_infected_limit=50050,
                                        model_property=model_property, lockdown_strength=80)

    optimize_SIR_model.optimize(time=15, interval=5, verbose=False)
    optimize_SEIR_model.optimize(time=15, interval=5, verbose=False)
    optimize_SIRD_model.optimize(time=15, interval=5, verbose=False)
    optimize_SEIRD_model.optimize(time=15, interval=5, verbose=False)
    optimize_SEIivIcvRDVIm_model.optimize(time=15, interval=5, verbose=False)

    Modelplot([optimize_SIR_model, optimize_SEIR_model, optimize_SIRD_model,
                optimize_SEIRD_model, optimize_SEIivIcvRDVIm_model], plot_type=1)