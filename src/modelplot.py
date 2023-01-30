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
import matplotlib.pyplot as plt
import re
from epidemiology import *


class Modelplot:
    """
    A class for handling different epidemiological post-simulation data visualization plots.

    A possible data visualization types.
    Type 1: 
        Ploting the model's different categories over time and population of multiple 
        given cities as subplot into one single plot. 
    Type 2:
        Ploting one city's different model's categories over time and population as 
        subplot into one single plot
    Type 3:
        Plotting one city's different model's categories over time and population in 
        different plot for different categores.

    Attributes
    ----------
    *model_object_list : arguments
        vector of arguments for passing multiple objects of class type Epidemiology
    plot_type        : int
        An integer variable for selecting the type of plot. (1,2,3)
    modelObejcts    : list
        A list of objects of class Epidemiology for ploting
    model_name       : list
        A list of all given model objects' respective name
    model_city_name  : list
        A list of all given cities, over which the epidemiological simulation had done
    model_dict      : dict
        A global dictionary for Epidemiology model's Initials name
    model_fig_size    : dict
        A global dictionary for figure size for different models

    Methods
    -------
    multi_city_one_plot(model_object : list)
        Plotting a model's different categoris as a subplot in one single plot 
        for multiple cities over time and population.
        Different plot for different Epidemiology models.

    one_city_multi_plot(model_object : list)
        Plotting a city's corresposing all models as subplot in a single plot. 
        Each city will have its respective plot.

    one_city_multi_category_plot(model_objects : list)
        Plotting a city's corresponding all models' category as subplot in a single plot. 
        Each cities' different model's category will have differnt plot.

    getTitle(model_name : str, c : int)
        Return the Model Title for the plot by looking the given model name and it's respective category
    getSubplotSize(model_name : str)
        Return the size of the corresponding size of the subplot by the given model. 
    """

    def __init__(self, model_object_list:list, plot_type:int):
        """
        Args:
            *model_object_list: the different model's object of class type Epidemiology
            plot_type: (int) type pf the plot for model post-simulation visualization [1,2,3]
        """
        self.plot_type = plot_type
        self.model_objects = []
        self.model_name = []
        self.model_city_name = []
        for i in model_object_list:
            self.model_objects.append(i) 
        self.model_city_name += list(self.model_objects[0].data_frame['place'])
        self.model_dict = {
                        "S": 'Susceptible', 'E': 'Exposed', 'I': 'Infected', 
                        'Iiv':'Infected Incomplete Vaccination', 'Icv':'Infected Complete Vaccination', 
                        'R': 'Recovered', 'D': 'Dead', 'V': 'Vaccinated', 'Im':'Immunity'
        }
        self.model_fig_size = {"SIR": 220, "SEIR": 220,"SIRD": 220, "SEIRD": 320, "SEIivIcvRDVIm": 420}
        self.model_R_index = {"SIR": 2, "SEIR": 3,"SIRD": 2, "SEIRD": 3, "SEIivIcvRDVIm": 4}
        self.model_I_index = {"SIR": 1, "SEIR": 2,"SIRD": 1, "SEIRD": 3}
        self.model_count = len(model_object_list)
        
        if plot_type == 1:
            self.multi_city_one_plot(self.model_objects)
        elif plot_type == 2:
            self.one_city_multi_plot(self.model_objects)
        elif plot_type == 3:
            self.one_city_multi_category_plot(self.model_objects)
        else:
            raise Exception("Invalid plot type.")

    def getTitle(self, model_name:str, c:int) -> str:
        """
        Get the title of a model.

        Args:
            model_name: (str) Name of the epidemiology model
            c: (int) model's corresponding different categories index

        Return:
            Title of the model as string data type.
        """
        
        if model_name == 'SEIivIcvRDVIm':
            if c == 0:
                return self.model_dict['S']
            elif c == 1:
                return self.model_dict['E']
            elif c == 2:
                return self.model_dict['Iiv']
            elif c == 3:
                return self.model_dict['Icv']
            elif c == 4:
                return self.model_dict['R']
            elif c == 5:
                return self.model_dict['D']
            elif c == 6:
                return self.model_dict['V']
            elif c == 7:
                return self.model_dict['Im']

        return self.model_dict[model_name[c]]
    
    def getSubplotSize(self, model_name: str) -> list:
        """
        Get the size of the subplot format.

        Args:
            model_name: (str) Name of the epidemiology model.
        Return:
            The size of the subplot as int data type.

        """
        return list(i for i in range(self.model_fig_size[model_name]+1, self.model_fig_size[model_name]+9))
    
        
    def multi_city_one_plot(self, model_objects:Epidemiology):
        """
            Plotting a model's different categoris as a subplot in one single plot 
            for multiple cities over time and population.
            Different plot for different Epidemiology models.

            Args:
                model_objects: list of instances of 'Epidemiology' class.
            Return:
                None.
        """
        model_count = len(self.model_objects)
        time_frame = [] 
        total_cities = []
        S = []
        for ti in range(model_count):
            time_frame += list([self.model_objects[ti].time_points])
            total_cities += list([self.model_objects[ti].total_cities])

            
        for model_index in range(model_count):
            plt.figure(figsize=(12, 12), edgecolor='black')
            plt.style.use('seaborn')
            subplotSize = self.getSubplotSize(self.model_objects[model_index].model_name)
            title = self.model_objects[model_index].model_name +" "+ self.model_objects[model_index].solver_type 
            plt.suptitle(title)
            count = 0
            model_category = (len(self.model_objects[model_index].model_name) 
                        if self.model_objects[model_index].model_name != 'SEIivIcvRDVIm' else 8)
            
            for category in range(model_category):
                for cityIndex in range(total_cities[model_index]):
                    
                    plt.subplot(subplotSize[count])
                    plt.plot(self.model_objects[model_index].output_sol[cityIndex][:, category],
                            label=self.model_objects[model_index].data_frame.place[cityIndex])
                    plt.xlabel('Time (days)')
                    plt.ylabel('Population')
                    plt.legend(fontsize='x-large', facecolor='white', fancybox=True,title="Place", title_fontsize='xx-large')
                    plt.grid(color='green', linestyle='--', linewidth=0.5)
                    plt.title(self.getTitle(self.model_objects[model_index].model_name, category), fontsize=20)

                count = count +1
                plt.legend(fontsize='x-large', facecolor='white', fancybox=True,title="Place", title_fontsize='xx-large')
                plt.locator_params(axis="x", integer=True, tight=True)

        plt.show()
    
    def one_city_multi_plot(self, model_objects:Epidemiology):
        """
            Plotting a city's corresposing all models as subplot in a single plot.
            Each city will have its respective plot.

            Args:
                model_objects : list of instances of 'Epidemiology' class.
            Return:
                None.
        """
        # self.df_row=self.model_objects[0].data_frame.shape[0]
        # TODO: generalize
            

        for model_index in self.model_objects:
            self.model_name.append(model_index.model_name)
        if(self.model_count == 1):
                row=1; col = 1
        elif(self.model_count == 2):
            row=1; col=2
        elif(self.model_count==3 or self.model_count==4):
            row=2; col=2
        elif(self.model_count==5):
            row=3; col=2

        data = []

        for model_index in self.model_objects:
            data.append(model_index.output_sol)

        c = 0

        for cityIndex in range(len(self.model_city_name)):
            x=1
            plt.figure(figsize=(10,10), edgecolor='black')
            plt.style.use('seaborn')
            plt.suptitle(self.model_city_name[c])
            
            for model_index in range(len(self.model_name)):       
                plt.subplot(row,col,x) 
                plt.plot(data[model_index][cityIndex])
                
                x += 1
                label = self.model_objects[model_index].model_name +" "+ self.model_objects[model_index].solver_name

                plt.title(label)
                if(self.model_name[model_index] == 'SIR'):
                    plt.legend(['S','I','R'])

                if(self.model_name[model_index] == 'SEIR'):
                    plt.legend(['S','E','I','R'])

                if(self.model_name[model_index] == 'SEIRD'):
                    plt.legend(['S','E','I','R','D'])
                    
                if(self.model_name[model_index] == 'SIRD'):
                    plt.legend(['S','I','R','D'])

                if(self.model_name[model_index] == 'SEIRD'):
                    plt.legend(['S','E','I','R','D'])
                
                if(self.model_name[model_index] == 'SEIivIcvRDVIm'):
                    plt.legend(['S', 'E', 'I_iv', 'I_cv', 'R', 'D', 'V', 'Im'])

                plt.xlabel('Time(days)')
                plt.ylabel('Population')

            c+=1
            plt.locator_params(axis="x", integer=True, tight=True)

        plt.show()
        
    def one_city_multi_category_plot(self, model_objects:Epidemiology):
        """
            Plotting a city's corresponding all models' category as subplot in a single plot. 
            Each cities' different model's category will have differnt plot.

            Args:
                model_objects: list of instances of 'Epidemiology' class.
            Return
                None.
        """

        plotStyle = ['o', '^', 's', 'v', '>', '<', 'd', 'D', 'h']
        plotCategory = {}
        infected = []

        for states in range(len(self.model_city_name)):
            plt.figure()
            plt.title(str(self.model_city_name[states] + " - " + self.model_dict['S']))
            
            for model_index in range(len(self.model_objects)):
                label = self.model_objects[model_index].model_name +" "+ self.model_objects[model_index].solver_name

                plt.plot(self.model_objects[model_index].output_sol[states][:,0], marker=plotStyle[model_index], 
                        label=label)
            
            plt.legend()
            plt.locator_params(axis="x", integer=True, tight=True)

        for states in range(len(self.model_city_name)):
            plt.figure()
            plt.title(str(self.model_city_name[states] + " - " + self.model_dict['I']))
            
            for model_index in range(len(self.model_objects)):
                label = self.model_objects[model_index].model_name +" "+ self.model_objects[model_index].solver_name

                if self.model_objects[model_index].model_name == 'SEIivIcvRDVIm':
                    infected = [sum(i) for i in zip(self.model_objects[model_index].output_sol[states][:,2], 
                                self.model_objects[model_index].output_sol[states][:,3])]
                    plt.plot(infected, marker=plotStyle[model_index], label=label)
                
                else:
                    plt.plot(
                        self.model_objects[model_index].output_sol[states][:,self.model_R_index[self.model_objects[model_index].model_name]], 
                        marker=plotStyle[model_index], label=label)

            plt.legend()
            plt.locator_params(axis="x", integer=True, tight=True)

        for states in range(len(self.model_city_name)):
            plt.figure()
            plt.title(str(self.model_city_name[states] + " - " + self.model_dict['R']))
            
            for m in range(len(self.model_objects)):
                label = self.model_objects[m].model_name +" "+ self.model_objects[m].solver_name
                plt.plot(self.model_objects[m].output_sol[states][:,self.model_R_index[self.model_objects[m].model_name]], 
                        marker=plotStyle[m], label=label)
            
            plt.legend()
            plt.locator_params(axis="x", integer=True, tight=True)

        plt.show()

class plot_model_GDP:

    def __init__(self, model_object_list: list):
        self.model_objects = []
        self.model_name = []
        for i in model_object_list:
            self.model_objects.append(i) 

        self.plot_GDP()

    def plot_GDP(self):
        for i in range(len(self.model_objects)):
            label = self.model_objects[i].model_name +" "+ self.model_objects[i].solver_name
            plt.plot(self.model_objects[i].max_list, label=label)

        plt.xlabel('Interval')
        plt.ylabel('GDP')
        plt.legend()

        plt.show()