import pandas as pd
import numpy as np
from models.RandomForestModel import RFModelWithFeatureSelection
from models.RandomForestModel import RFModel
import matplotlib.pyplot as plt
from models.GeneticAlgorithm import GeneticAlgorithmFS
from models.SVMModel import SVMModel
import matplotlib.pyplot as plt
from models.NeuralNetworkModel import NNModel
from tqdm import tqdm
import time

base_directory = "../data/SETAP PROCESS DATA CORRECT AS FIE2016/" 

dfs = []

for i in range(1,12):
    file_name = base_directory + f"setapProcessT{i}.csv"
    df = pd.read_csv(file_name, header=1)

    # df['SetapNum'] = f'T{i}'
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
dependent_variable = "SE Process grade"
total_num_variables = combined_df.shape[1]

def task1_evaluate_models(df, target_column, max_features):
    metrics = {
        "RF_with_fs": {"precision": [], "recall": [], "accuracy": [], "f1": [], "speed": [], "Number of Features": []},
        "RF_without_fs": {"precision": [], "recall": [], "accuracy": [], "f1": [], "speed": [], "Number of Features": []},
        "NN_with_fs": {"precision": [], "recall": [], "accuracy": [], "f1": [], "speed": [], "Number of Features": []},
        "NN_without_fs": {"precision": [], "recall": [], "accuracy": [], "f1": [], "speed": [], "Number of Features": []}
    }

    feature_range = range(1, max_features + 1)
    
    for n_features in tqdm(feature_range, desc='Training models'):
        # Random Forest with feature selection
        start_time = time.time()
        model_with_fs = RFModelWithFeatureSelection(df=df, target_column=target_column, n_features=n_features)
        result_rf_with_fs = model_with_fs.train()
        execution_time = time.time() - start_time
        metrics["RF_with_fs"]["precision"].append(result_rf_with_fs["test_precision"])
        metrics["RF_with_fs"]["recall"].append(result_rf_with_fs["test_recall"])
        metrics["RF_with_fs"]["accuracy"].append(result_rf_with_fs["test_accuracy"])
        metrics["RF_with_fs"]["f1"].append(result_rf_with_fs["test_f1"])
        metrics["RF_with_fs"]["speed"].append(execution_time)
        metrics["RF_with_fs"]["Number of Features"].append(n_features)


        # Random Forest without feature selection
        start_time = time.time()
        model_without_fs = RFModel(df=df, target_column=target_column)
        result_rf_without_fs = model_without_fs.train()
        execution_time = time.time() - start_time
        metrics["RF_without_fs"]["precision"].append(result_rf_without_fs["test_precision"])
        metrics["RF_without_fs"]["recall"].append(result_rf_without_fs["test_recall"])
        metrics["RF_without_fs"]["accuracy"].append(result_rf_without_fs["test_accuracy"])
        metrics["RF_without_fs"]["f1"].append(result_rf_without_fs["test_f1"])
        metrics["RF_without_fs"]["speed"].append(execution_time)
        metrics["RF_without_fs"]["Number of Features"].append(total_num_variables)

        # Neural Network with feature selection
        # Note: Assuming NNModel has similar methods to return metrics & execution time
        start_time = time.time()
        model_NN_with_fs = NNModel(data=df, target_column=target_column, n_features=n_features)
        result_nn_with_fs = model_NN_with_fs.train_with_SK_feature_selection()
        execution_time = time.time() - start_time
        metrics["NN_with_fs"]["precision"].append(result_nn_with_fs["test_precision"])
        metrics["NN_with_fs"]["recall"].append(result_nn_with_fs["test_recall"])
        metrics["NN_with_fs"]["accuracy"].append(result_nn_with_fs["test_accuracy"])
        metrics["NN_with_fs"]["f1"].append(result_nn_with_fs["test_f1"])
        metrics["NN_with_fs"]["speed"].append(execution_time)
        metrics["NN_with_fs"]["Number of Features"].append(n_features)

        # Neural Network without feature selection
        start_time = time.time()
        model_NN_without_fs = NNModel(data=df, target_column=target_column)
        result_nn_without_fs = model_NN_without_fs.train_and_evaluate()
        execution_time = time.time() - start_time
        metrics["NN_without_fs"]["precision"].append(result_nn_without_fs["test_precision"])
        metrics["NN_without_fs"]["recall"].append(result_nn_without_fs["test_recall"])
        metrics["NN_without_fs"]["accuracy"].append(result_nn_without_fs["test_accuracy"])
        metrics["NN_without_fs"]["f1"].append(result_nn_without_fs["test_f1"])
        metrics["NN_without_fs"]["speed"].append(execution_time)
        metrics["NN_without_fs"]["Number of Features"].append(total_num_variables)

    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for idx, metric in enumerate(["precision", "recall", "accuracy", "f1"]):
        ax = axs[idx//3, idx%3]
        for key in metrics:
            ax.plot(feature_range, metrics[key][metric], label=key)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Impact of Feature Selection on {metric.capitalize()}')
        ax.legend()

    # Speed Visualization
    ax = axs[1, 2]
    for key in metrics:
        ax.plot(feature_range, metrics[key]["speed"], label=key)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Model Training and Evaluation Time')
    ax.legend()

    plt.tight_layout()
    plt.show()

# Assuming combined_df is your DataFrame and dependent_variable is your target column
num_variables = combined_df.shape[1] - 1
task1_evaluate_models(combined_df, dependent_variable, max_features=num_variables)



