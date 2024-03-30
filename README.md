# Assignmeng 2 Data Science for Software Engineering

To discover the strategic nuances between the performance and the computational complexity of different feature selection methods.

## Prerequisites

Python 3

## Getting Started

Please follow the following instructions in order to run the scripts

To start the virtual environment, run `python3 -m venv myenv ` in your terminal.

To activate the virtual environment, run `source myenv/bin/active` in your terminal.

To install all the dependencies, run `pip3 install -r requirement.txt` in your terminal

If you are using MacOS, you can take advantage of the predeclared script `running_all.sh` . This script will automatically run all three tasks. Start the script by running 

    sh running_all.sh

If you are on other OS, please use the following command and replace N with the task you want to run:
    
    cd src
    python3 task_N_main.py

## What to expect

`src/task_1_main.py` : Using Sklearn Built-in feature learning to train Random Forest and Neural Network. Explore the relationship between training time, training result(f1, accuracy, precision, recall), and number of feature selected. 

`src/task_2_main.py`: Using Genetic Algothorism to conduct feature selection. Random Forest is used to preform fitness evaluation. Explored the relationship between best fitness of feature selected and number of generations(10, 30, 50, 80, 100) ran. 

`src/task_3_main.py`: Using Simulated Annealing to conduct feature selection. Both Random Forest and Neural Network are used in fitness evaluation. Explored the relationship between best fitness of feature selected, model used to conduct fitness evaluation, and number of iterations(50, 100, 250, 500, 750, 875, 1000) ran. 

Please note the time it takes to run the three task scripts varies.

`src/task_1_main.py` : 6 min

`src/task_2_main.py` : 80 min

`src/task_3_main.py` : 40 min

I also provided estimate time for each round of training to finish upon running of the script. 

