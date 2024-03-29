import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from models.SVMModel import SVMModel
from models.RandomForestModel import RFModelWithFeatureSelection
from models.NeuralNetworkModel import NNModel
from tqdm import tqdm

class SimulatedAnnealingFeatureSelection:
    def __init__(self, data, target_column, model_choice='NeuralNetwork', initial_temperature=100, cooling_rate=0.95, n_iterations=1000):
        self.data = data
        self.target_column = target_column
        self.model_choice = model_choice  # Add model choice parameter
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.n_iterations = n_iterations
        self.temperature = initial_temperature
        self.feature_names = data.drop(columns=[target_column]).columns
        self.best_solution = None
        self.best_fitness = -np.inf
        self.models = {
            'RFModelWithFeatureSelection': RFModelWithFeatureSelection(data, target_column),
            'NeuralNetwork': NNModel(data, target_column)
        }
        # Ensure model exists
        if model_choice not in self.models:
            raise ValueError(f"Model choice '{self.model_choice}' is not supported.")
        

    def evaluate_fitness_SVM(self, solution):
        selected_features = [self.feature_names[i] for i in range(len(solution)) if solution[i]]
        if not selected_features:  # If no features are selected, return a low fitness score
            return 0
        test_f1 = self.svm_model.train_with_selected_features(selected_features)
        return test_f1

    def evaluate_fitness_RF(self, solution):
        # Convert the binary solution (True/False array) into selected feature names
        selected_features = [self.feature_names[i] for i in range(len(solution)) if solution[i]]

        if not selected_features:  # If no features are selected, return a low fitness score
            return 0

        # Now, call the RF model's training method with the correct list of selected feature names
        test_f1 = self.rf_model.train_with_selected_features(selected_features)
        return test_f1
    
    def evaluate_fitness_NN(self, solution):
        """
        Evaluates the fitness of a solution using the Neural Network model.
        """
        selected_features = [self.feature_names[i] for i in range(len(solution)) if solution[i]]
        if not selected_features:  # If no features are selected, return a low fitness score
            return 0
        # Use the NN model to train with the selected features and calculate the F1 score
        resultNN = self.nn_model.train_with_selected_features(selected_features)
        return resultNN
    def evaluate_fitness(self, solution):
        model = self.models[self.model_choice]

        # Convert the binary solution (True/False array) to the appropriate format
        if self.model_choice == 'NeuralNetwork':
            # For Neural Network, convert to actual feature names
            selected_features = [self.feature_names[i] for i in range(len(solution)) if solution[i]]
        else:
            # For other models, use the boolean array directly
            selected_features = solution

        if not any(selected_features):  # If no features are selected, return a low fitness score
            return 0

        # Adapt method call based on input type (feature names for NN, boolean mask for others)
        if self.model_choice == 'NeuralNetwork':
            resultDict = model.train_with_selected_features(selected_features)
        else:
            # Convert selected features from boolean mask to indices for RF
            selected_indices = [i for i, selected in enumerate(solution) if selected]
            resultDict = model.train_with_selected_features(selected_indices)

        return resultDict['test_f1']

    def generate_neighbor(self, solution):
        neighbor = solution.copy()
        # Randomly add or remove a feature to create a neighbor solution
        index = np.random.randint(len(neighbor))
        neighbor[index] = not neighbor[index]
        return neighbor

    def acceptance_probability(self, old_fitness, new_fitness):
        if new_fitness > old_fitness:
            return 1.0
        else:
            return np.exp((new_fitness - old_fitness) / self.temperature)

    def run(self):
        # Initialize with a random solution
        current_solution = np.random.choice([True, False], size=len(self.feature_names))
        current_fitness = self.evaluate_fitness(self.feature_names[current_solution])

        for iteration in tqdm(range(self.n_iterations), desc='Simulated Annealing Progress'):
            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness = self.evaluate_fitness(self.feature_names[neighbor])
 
            if self.acceptance_probability(current_fitness, neighbor_fitness) > np.random.rand():
                current_solution = neighbor
                current_fitness = neighbor_fitness
            
            if neighbor_fitness > self.best_fitness:
                self.best_solution = neighbor
                self.best_fitness = neighbor_fitness
            
            self.temperature *= self.cooling_rate  # Cool down
            # if iteration % 10 == 0:
                # print(f"Current iteration {iteration}, Current Fitness Score: {current_fitness}")
        
        selected_features = self.get_selected_features(self.best_solution)
        return self.best_solution, self.best_fitness, selected_features

    def get_selected_features(self, solution):
        # Initialize an empty list to hold the names of selected features
        selected_features = []
        
        # Iterate through each bit in the individual
        for i in range(len(solution)):
            # Check if the bit is set to 1, meaning the feature is selected
            if solution[i] == 1:
                # If the feature is selected, append its name to the list
                feature_name = self.feature_names[i]
                selected_features.append(feature_name)
        
        # Return the list of selected feature names
        return selected_features
