import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.GeneticAlgorithm import GeneticAlgorithmFS
from models.NeuralNetworkModel import NNModel  # Assuming this is your NN model
from time import time

base_directory = "../data/SETAP PROCESS DATA CORRECT AS FIE2016/" 
dfs = []

for i in range(1, 12):
    file_name = base_directory + f"setapProcessT{i}.csv"
    df = pd.read_csv(file_name, header=1)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
dependent_variable = "SE Process grade"
# generations_list = [10, 30]
generations_list = [10, 30, 50, 80, 100]
# model_choices = ['RFModelWithFeatureSelection', 'NeuralNetwork']  # Added 'NNModel' to the list of models
model_choices = ['RFModelWithFeatureSelection']  # Added 'NNModel' to the list of models
metrics = {model: {"generations": [], "num_features": [], "speed": [], "f1": []} for model in model_choices}

for model_choice in model_choices:
    print(f"Running GA FS with model: {model_choice}")
    for generations in generations_list:
        print("\033[32m"+ "_______________________________________________________"+ "\033[0m")
        print("\033[32m"+ f"Generating Result for {generations} generations with model: {model_choice}"+ "\033[0m")
        print("\033[32m"+"See estimated finish time for this round below: \n"+ "\033[0m")
        start_time = time()
        ga_fs = GeneticAlgorithmFS(combined_df, dependent_variable, model_choice=model_choice, n_generations=generations)
        best_individual, best_features, final_best_fitness = ga_fs.run()

        execution_time = time() - start_time

        metrics[model_choice]["generations"].append(generations)
        metrics[model_choice]["num_features"].append(len(best_features))
        metrics[model_choice]["speed"].append(execution_time/int(generations))
        metrics[model_choice]["f1"].append(final_best_fitness)

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Number of Features Selected
for model_choice in model_choices:
    axs[0].plot(metrics[model_choice]["generations"], metrics[model_choice]["num_features"], marker='o', label=model_choice)
axs[0].set_title("Number of Features Selected")
axs[0].set_xlabel("Generations")
axs[0].set_ylabel("Number of Features")
axs[0].legend()

# Execution Time
for model_choice in model_choices:
    axs[1].plot(metrics[model_choice]["generations"], metrics[model_choice]["speed"], marker='o', label=model_choice)
axs[1].set_title("Execution Speed")
axs[1].set_xlabel("Generations")
axs[1].set_ylabel("Time (seconds/generation)")
axs[1].legend()

# F1 Score
for model_choice in model_choices:
    axs[2].plot(metrics[model_choice]["generations"], metrics[model_choice]["f1"], marker='o', label=model_choice)
axs[2].set_title("F1 Score")
axs[2].set_xlabel("Generations")
axs[2].set_ylabel("F1 Score")
axs[2].legend()

plt.tight_layout()
plt.show()
