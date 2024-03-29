import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from models.SimulatedAnnealingFeatureSelection import SimulatedAnnealingFeatureSelection

# Assuming the rest of the setup is the same, including loading data
base_directory = "../data/SETAP PROCESS DATA CORRECT AS FIE2016/" 
dfs = []
for i in range(1, 12):
    file_name = base_directory + f"setapProcessT{i}.csv"
    df = pd.read_csv(file_name, header=1)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
dependent_variable = "SE Process grade"

model_choices = ['RFModelWithFeatureSelection', 'NeuralNetwork']  # List of model choices for comparison
iterations_list = [50, 100, 250, 500, 750, 875, 1000]
# iterations_list = [50, 100]
metrics = {model: {"iterations": [], "num_features": [], "speed": [], "best_fitness": []} for model in model_choices}

for model_choice in model_choices:
    print(f"\033[32mRunning SA FS with model: {model_choice}\033[0m")
    for iterations in iterations_list:
        print("\033[32m"+ "_______________________________________________________"+ "\033[0m")
        print(f"\033[32mGenerating Result for {iterations} Iterations with model: {model_choice}\033[0m")
        print("\033[32m"+"See estimated finish time for this round below: \n"+ "\033[0m")
        start_time = time()
        sa_fs = SimulatedAnnealingFeatureSelection(combined_df, dependent_variable, model_choice=model_choice, n_iterations=iterations)
        best_solution, best_fitness, best_features = sa_fs.run()

        execution_time = time() - start_time

        metrics[model_choice]["iterations"].append(iterations)
        metrics[model_choice]["num_features"].append(len(best_features))
        metrics[model_choice]["speed"].append(execution_time/int(iterations))
        metrics[model_choice]["best_fitness"].append(best_fitness)

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Number of Features Selected
for model_choice in model_choices:
    axs[0].plot(metrics[model_choice]["iterations"], metrics[model_choice]["num_features"], marker='o', label=model_choice)
axs[0].set_title("Number of Features Selected")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Number of Features")
axs[0].legend()

# Execution Time
for model_choice in model_choices:
    axs[1].plot(metrics[model_choice]["iterations"], metrics[model_choice]["speed"], marker='o', label=model_choice)
axs[1].set_title("Execution Speed")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Time (seconds/iterations)")
axs[1].legend()

# Best Fitness Score
for model_choice in model_choices:
    axs[2].plot(metrics[model_choice]["iterations"], metrics[model_choice]["best_fitness"], marker='o', label=model_choice)
axs[2].set_title("Best Fitness Score")
axs[2].set_xlabel("Iterations")
axs[2].set_ylabel("Fitness Score")
axs[2].legend()

plt.tight_layout()
plt.show()
