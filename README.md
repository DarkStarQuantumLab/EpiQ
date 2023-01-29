# Epidemiological Solutions with Quantum Annealing - EpiQ


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

## License

This work is licensed under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0) and is owned by [DarkStarQuantumLab, Inc.](https://github.com/DarkStarQuantumLab). 
