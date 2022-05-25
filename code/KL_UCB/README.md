## Decentralized Multi-Armed Bandit Can Outperform Classic Upper Confidence Bound

Implementation of the Decentralized KL-UCB and Decentralized UCB1 algorithms, and code used to generate simulations found in the paper.

### Requirements

A requirements.txt file has been provided for setup of dependencies. To use this file (Anaconda environment is used as an example), navigate in a terminal to the directory containing requirements.txt and execute the following commands, replacing `<env_name>` with the name of your new virtual environment:

`$ conda create -n <env_name> python=3.7`

`$ conda activate <env_name>`

`$ pip install -r requirements.txt`

If this does not work, or you would prefer not to go the virtual environment route, just be aware that to run the code you need Python 3 (development was completed with Python 3.7+) installed along with the following packages:
- [networkx](https://networkx.org/)
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)

NOTE: All Jupyter Notebook files use multiprocessing to run multiple trials simultaneously. They are verified to run on macOS and Unix platforms, but Windows is known to have issues with Python multiprocessing and has not been tested.

### Usage

`Dist_KL_UCB.py` and `Dist_UCB1.py` contain the implementations for the Distributed KL-UCB and Distributed UCB1 algorithms for the main paper simulations.

`Dist_KL_UCB_Small_Graphs.py` and `Dist_UCB1_Small_Graphs.py` contain the implementations for the Distributed KL-UCB and Distributed UCB1 algorithms for the supplemental simulations

Each of these files has commented test run lines that can be used to test the implementations by uncommenting them and typing `python filename.py` in the terminal, replacing `filename.py` with the name of the file. Running the files will output a plot of the regrets of the best and worst performing agents.

Jupyter Notebook can be used to run the following:

`Dist_vs_Single_KL_UCB.ipynb` can be ran to obtain Figure 1.

`Dist_vs_Single_UCB1.ipynb` can be ran to obtain Figure 2.

`Dist_KL_UCB_vs_Dist_UCB1.ipynb` can be ran to obtain Figure 3.

`Dist_vs_Single_KL_UCB_Beta.ipynb` can be ran to obtain Figure 4.

`Dist_vs_Single_UCB1_Beta.ipynb` can be ran to obtain Figure 5.

`Dist_KL_UCB_vs_Dist_UCB1_Beta.ipynb` can be ran to obtain Figure 6.

`Dist_vs_Single_KL_UCB_Small_Graphs.ipynb` can be ran to obtain Figure 8.

`Dist_vs_Single_UCB1_Small_Graphs.ipynb` can be ran to obtain Figure 9.


