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

from src.epidemiology import *
from src.modelplot import *
import pandas as pd

if __name__ == "__main__":
	# load data
	dataFrame=pd.read_csv('data/data.csv') 
	timePoints=[x for x in range(1,50)]	
	distanceDataFrame = pd.read_csv('data/distance.csv', index_col = 0, header =0)

	
	model_SIR = Epidemiology(model_name='SIR', time_points=timePoints, data_frame=dataFrame, 
						distance_dataframe=distanceDataFrame, solver_type="classical", show_dataset=True )

	model_SIRD = Epidemiology(model_name='SIRD', time_points=timePoints, data_frame=dataFrame, 
						distance_dataframe=distanceDataFrame, solver_type="classical", show_dataset=True )

	model_SEIR = Epidemiology(model_name='SEIR', time_points=timePoints, data_frame=dataFrame, 
						distance_dataframe=distanceDataFrame, solver_type="classical", show_dataset=True )

	model_SEIRD = Epidemiology(model_name='SEIRD', solver_type='classical', time_points=timePoints, data_frame=dataFrame, 
						distance_dataframe=distanceDataFrame, show_dataset=True )

	model_SEIRDV = Epidemiology(model_name='SEIRDV', solver_type='classical', time_points=timePoints, data_frame=dataFrame, 
						distance_dataframe=distanceDataFrame, show_dataset=True )

	Modelplot([model_SIR, model_SIRD, model_SEIR, model_SEIRD, model_SEIRDV], plot_type = 1)