import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from models.RandomForestModel import RFModelWithFeatureSelection
from models.NeuralNetworkModel import NNModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class Chromosome:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        # Initialize chromosome with random feature selection
        self.genes = np.random.choice([0, 1], size=(len(df.columns) - 1,))
        self.fitness = -1

    def compute_fitness(self):
        # Convert gene sequence to list of selected feature indices
        selected_features = np.where(self.genes == 1)[0]

        if len(selected_features) == 0:  # Ensure there's at least one feature
            self.fitness = 0
        else:
            model = RFModelWithFeatureSelection(self.df, self.target_column)
            _, self.fitness = model.train_with_selected_features(selected_features)

        return self.fitness


class GeneticAlgorithmFS:
    def __init__(self, data, target_variable, model_choice='RFModelWithFeatureSelection', population_size=50, crossover_rate=0.8, mutation_rate=0.1, n_generations=100, tournament_size=3, elitism_size=2):
        self.data = data
        self.target_variable = target_variable
        self.model_choice = model_choice
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.feature_names = [col for col in data.columns if col != target_variable]
        self.population = []
        self.best_individuals_per_generation = []  # Added this line to store best per generation
        self.best_score_per_generation = []

    def initialize_population(self):
        n_features = len(self.data.columns) - 1  # Exclude the target variable
        self.population = []  # Starting with an empty population
        for _ in range(self.population_size):
            individual = np.random.randint(2, size=n_features)  # Either 0 or 1, with 0 being not selected and 1 being selected
            # print(individual)
            self.population.append(individual)  # Add the individual to the population: each individual will be a lits of [1,0,0,1,....] representing if selection of the features
            # now we have a population list of different 

    def calculate_fitness(self, individual):
        # Dynamically select and use the model based on the initialization parameter
        if self.model_choice == 'RFModelWithFeatureSelection':
            model = RFModelWithFeatureSelection(self.data, self.target_variable)
            # For RFModelWithFeatureSelection, we assume it takes a boolean mask directly
            result = model.train_with_selected_features(individual.astype(bool))
        elif self.model_choice == 'NeuralNetwork':
            model = NNModel(self.data, self.target_variable)  # Initialize the NNModel instance
            # For NeuralNetwork, we convert the boolean mask to actual feature names
            selected_features = [self.feature_names[i] for i, selected in enumerate(individual) if selected]
            result = model.train_with_selected_features(selected_features)
        else:
            raise ValueError(f"Model choice '{self.model_choice}' is not supported.")
        
        # Assuming both models return a dictionary with a "test_f1" key for the F1 score
        return result["test_f1"]


    def selection(self):
        # We are selecting parents for the next generation
        # each tournament refers to a completetion between different populations(different kinds of selected features)
        winners = []  # This will hold the winners of the tournaments
        for _ in range(self.population_size):
            # Randomly pick individuals for the tournament
            tournament = []  # A list to track tournament participants
            for _ in range(self.tournament_size):
                participant_index = np.random.randint(len(self.population))
                tournament.append(self.population[participant_index])
            
            # Evaluate fitness for each participant
            tournament_fitness = []
            for individual in tournament:
                fitness = self.calculate_fitness(individual)  # Calculate fitness
                tournament_fitness.append(fitness)
            
            # Determine the winner of the tournament
            winner_index = np.argmax(tournament_fitness)  # Index of the highest fitness
            winner = tournament[winner_index]  # The winning individual
            winners.append(winner)  # Add winner to the list of selected individuals
        
        # now we will have a list of winning populations(which are sets of the highest performing Combination of features)
        return winners  # Return the selected individuals

    def crossover(self, parent1, parent2):
        # we will randomly cross the selected features from both parents. Basiclly combining their "DNA"
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(parent1) - 1)  # Avoid trivial crossover
            return np.concatenate((parent1[:point], parent2[point:]))
        return parent1.copy()  # Return a copy to avoid mutation on original parent


    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]

    def evolve_population(self):
        # Main loop for evolving the population through selection, crossover, and mutation
        new_population = self.selection()
        for i in range(0, len(new_population), 2):
            parent1 = new_population[i]
            parent2 = new_population[i + 1] if i + 1 < len(new_population) else new_population[0]
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            self.mutate(child1)
            self.mutate(child2)
            new_population[i] = child1
            if i + 1 < len(new_population):
                new_population[i + 1] = child2
        self.population = new_population
    
    def get_selected_features(self, individual):
        # Initialize an empty list to hold the names of selected features
        selected_features = []
        
        # Iterate through each bit in the individual
        for i in range(len(individual)):
            # Check if the bit is set to 1, meaning the feature is selected
            if individual[i] == 1:
                # If the feature is selected, append its name to the list
                feature_name = self.feature_names[i]
                selected_features.append(feature_name)
        
        # Return the list of selected feature names
        return selected_features

    def run(self):
        self.initialize_population()
        self.best_individuals_per_generation = []  # To store the best individual of each generation

        # print("Generation | Best Fitness Score")
        # print("--------------------------------")

        for generation in tqdm(range(self.n_generations), desc='Evolving Generations'):
            self.evolve_population()
            
            # Finding the best individual of the current generation
            # Calculate fitness for each individual in the population
            print("Getting fitness score for all population")
            fitness_scores = [self.calculate_fitness(individual) for individual in self.population]

            # Find the index of the individual with the best fitness
            best_index = np.argmax(fitness_scores)
            print("Done")
            # Retrieve the best individual and its fitness using the best index
            best_of_generation = self.population[best_index]
            best_fitness = fitness_scores[best_index]

            # Storing the best individual for possible later use or analysis
            self.best_individuals_per_generation.append(best_of_generation)
            
            # Calculate and print the fitness of the best individual of the current generation
            # best_fitness = self.calculate_fitness(best_of_generation)
            self.best_score_per_generation.append(best_fitness)
            # print(f"{generation + 1:10} | {best_fitness:.4f}")

        # After all generations have been processed, find the overall best individual
        self.best_individual = max(self.population, key=self.calculate_fitness)
        
        # Calculate fitness for the best individual across all generations
        final_best_fitness = self.calculate_fitness(self.best_individual)
        
        # Get the list of features represented by the best individual
        best_features = self.get_selected_features(self.best_individual)

        # print(f"\nFinal Best Fitness Score: {final_best_fitness:.4f}")

        # Return both the binary representation and the list of features of the best individual
        return self.best_individual, best_features, final_best_fitness

    
    # debug function
    def visualize_best_feature_selection(self):
        # Aggregate feature selections
        selection_counts = np.sum(self.best_individuals_per_generation, axis=0)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(self.feature_names, selection_counts)
        plt.xlabel('Features')
        plt.ylabel('Selection Count')
        plt.title('Selection Frequency of Features Across Generations')
        plt.xticks(rotation=45)
        plt.show()

    def plot_best_fitness_scores(self):
        plt.figure(figsize=(10, 6))
        print(self.best_score_per_generation)
        plt.plot(self.best_score_per_generation, marker='o', linestyle='-', color='b')
        plt.title('Best Fitness Score Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness Score')
        plt.grid(True)
        plt.show()


    # def visualize_feature_selection_over_generations(self):
    #     # Create a binary matrix indicating feature selection in each generation
    #     selection_matrix = np.zeros((len(self.best_features_per_generation), len(self.feature_names)))
        
    #     for gen_index, features in enumerate(self.best_features_per_generation):
    #         for feature in features:
    #             feature_index = self.feature_names.index(feature)
    #             selection_matrix[gen_index, feature_index] = 1  # Mark the feature as selected
        
    #     # Plot the heatmap
    #     plt.figure(figsize=(12, 8))
    #     sns.heatmap(selection_matrix, cmap="YlGnBu", xticklabels=self.feature_names, yticklabels=False)
    #     plt.xlabel("Features")
    #     plt.ylabel("Generation")
    #     plt.title("Feature Selection Over Generations")
    #     plt.show()

