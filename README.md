#  EpiQ - Epidemiological Solutions with Quantum Annealing and Quantum Machine Learning
As COVID-19 examplified, the impact of a pandemic on global socioeconomics can be dramatic. This project uses first generation quantum processors, in particualr, quantum annealers, in the design of a phase-wise, optimal lockdown schedule among cities so that some of cities can stay open while minimizing the rate of deaths and maximizing the number of beds available in hospitals. It is observed that the lock-down schedule proposed by the quantum annealer is optimal compared to the one proposed by a classical one.  

The state-of-the-art tool used in this project is the Quantum Support Vecotr Machine, an instance of a Quantum Machine Learning algorithm. 

## Installation

For a local installation, ideally in a virtual environment, run:

    pip install -r requirements.txt

## Epidemiological Models

The following [epidemiological models](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) are available:

- SIR
- SIRD
- SEIR
- SEIRD
- SEIivlcvRDVIm

## Lockdown Management (Knapsack Problem)

The recommendation upon which a cities is recommended for lockdown is given by the solution of [the knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem). The algorithm takes into the consideration such parameters as the number of population in the city, the number of infected population, GDP of the city, the number of available places in hospitals. 
The knapsack could be solved on a classical computer (CPU), on the physical quantum hardware (QPU) available via [DWave Leap cloud services](https://www.dwavesys.com/solutions-and-products/cloud-platform/), or as the simulated annealing algorithm. 

An example the lockdown recommendation is depicted below.
| City // Days Elapsed |  0     | 5      |   10   |
|----------------------|--------|--------|--------|
| ``city 1``           | open   | closed |closed  |
| ``city 2``           | open   | closed |closed  |
| ``city 3``           | open   | closed |closed  |


## Infection Prediction (Quantum Support Vector Machine)

The QSVM makes predictions infected/uninfected based on the symptomes and patient's age.

The QSVM could be executed on a physical quantum annealing device or as the simulated annealing algorithm locally. 

## Data
A sample of data points for training is available in the [data](https://github.com/DarkStarQuantumLab/Epidemiological-Solutions-on-Quantum-Annealing/tree/main/data) subdirectory. 

- [Covid_Sym2.csv](https://github.com/DarkStarQuantumLab/Epidemiological-Solutions-on-Quantum-Annealing/blob/main/data/Covid_Sym2.csv) is a file with COVID-19 symptomes for the QSVM trainig.
- [data.csv](https://github.com/DarkStarQuantumLab/Epidemiological-Solutions-on-Quantum-Annealing/blob/main/data/data.csv) contains data points about cities (population, infected, recovered, GDP, etc.) 
- [distance.csv](https://github.com/DarkStarQuantumLab/Epidemiological-Solutions-on-Quantum-Annealing/blob/main/data/distance.csv) is the distances between cities.

## Examples
Some tutorial to get started with lockdown management.

[Epidemiology Models](https://github.com/DarkStarQuantumLab/Epidemiological-Solutions-with-Quantum-Annealing/blob/main/examples/Epidemiology%20Models.ipynb)

[Knapsack](https://github.com/DarkStarQuantumLab/Epidemiological-Solutions-with-Quantum-Annealing/blob/main/examples/Knapsack.ipynb)

[QSVM](https://github.com/DarkStarQuantumLab/Epidemiological-Solutions-with-Quantum-Annealing/blob/main/examples/QSVM.ipynb)


## Disclamer

The code in this repository is distributed on an "AS IS" basis, without any warranties or conditions of any kind. 

The code was tested on a classical CPU and DWave Quantum Anealer hardwares available via AWS Braket prior to November 18th 2022. To submit the code to a quantum hardware after November 18th 2022, a DWave's cloud API is required. More information could be found in the [dwave-cloud-client](https://docs.ocean.dwavesys.com/en/stable/docs_cloud/sdk_index.html) resource. 

To submit problems to Leap solvers, use a ``dwave-system``solver. More details can be found in the
[Ocean documentation](https://docs.ocean.dwavesys.com/en/stable/index.html).

## License

This work is licensed under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0) and is owned by [DarkStarQuantumLab, Inc.](https://github.com/DarkStarQuantumLab). 
