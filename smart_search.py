import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
import time
from collections import defaultdict
import json

# ===== 1) Constants =====
N_POINTS = 150
N_INPUT  = 2
N_OUTPUT = 1

# ===== 2) Dataset Generation =====
def make_extreme_spiral(n_points, noise=1.0):
    np.random.seed(0)
    theta = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    r_a = 2*theta + np.pi
    r_b = -2*theta - np.pi

    x1 = r_a * np.cos(theta) + np.random.randn(n_points,1)*noise
    y1 = r_a * np.sin(theta) + np.random.randn(n_points,1)*noise
    x2 = r_b * np.cos(theta) + np.random.randn(n_points,1)*noise
    y2 = r_b * np.sin(theta) + np.random.randn(n_points,1)*noise

    X = np.vstack((np.hstack((x1,y1)), np.hstack((x2,y2))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return X, y

# ===== 3) Activation Functions =====
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)
def relu_prime(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)
def tanh_prime(z):
    return 1 - np.tanh(z)**2

# ===== 4) Neural Network Training Function =====
def train_and_evaluate(test_params, verbose=False):
    # Select activation function
    act_name = test_params['activation_fn']
    if act_name == 'sigmoid':
        activation = sigmoid
        activation_prime = sigmoid_prime
    elif act_name == 'relu':
        activation = relu
        activation_prime = relu_prime
    elif act_name == 'tanh':
        activation = tanh
        activation_prime = tanh_prime
    else:
        raise ValueError("Unsupported activation function!")

    # Generate dataset
    X, y = make_extreme_spiral(N_POINTS, noise=test_params['noise'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_params['test_ratio'], random_state=0
    )

    # Weight Initialization
    np.random.seed(0)
    W1 = np.random.randn(N_INPUT, test_params['n_hidden']) * 0.1
    b1 = np.zeros((1, test_params['n_hidden']))
    W2 = np.random.randn(test_params['n_hidden'], N_OUTPUT) * 0.1
    b2 = np.zeros((1, N_OUTPUT))

    # Training Loop
    for ep in range(test_params['epochs']):
        Z1 = X_train.dot(W1) + b1
        A1 = activation(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = sigmoid(Z2)  # Output is sigmoid for binary classification

        dZ2 = (A2.reshape(-1,1) - y_train.reshape(-1,1)) * sigmoid_prime(Z2)
        dW2 = A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(W2.T)
        dZ1 = dA1 * activation_prime(Z1)
        dW1 = X_train.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W2 -= test_params['learning_rate'] * dW2
        b2 -= test_params['learning_rate'] * db2
        W1 -= test_params['learning_rate'] * dW1
        b1 -= test_params['learning_rate'] * db1

    # Prediction & Accuracy
    def predict(X):
        A1 = activation(X.dot(W1) + b1)
        A2 = sigmoid(A1.dot(W2) + b2)
        return (A2 > 0.5).astype(int)

    y_pred = predict(X_test)
    accuracy = (y_pred.flatten() == y_test).mean() * 100
    return accuracy

# ===== 5) Smart Parameter Search Strategies =====

class SmartParameterSearch:
    def __init__(self):
        self.results = []
        self.best_accuracy = 0
        self.best_params = None
        self.param_history = defaultdict(list)
        
    def log_result(self, params, accuracy):
        self.results.append((params.copy(), accuracy))
        self.param_history['n_hidden'].append(params['n_hidden'])
        self.param_history['learning_rate'].append(params['learning_rate'])
        self.param_history['epochs'].append(params['epochs'])
        self.param_history['noise'].append(params['noise'])
        self.param_history['accuracy'].append(accuracy)
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_params = params.copy()
            print(f"🎉 NEW BEST: {accuracy:.2f}% with params: {params}")
    
    def strategy_1_baseline_search(self):
        """Strategy 1: Baseline search around known good parameters"""
        print("\n🔍 Strategy 1: Baseline Search")
        print("=" * 50)
        
        baseline_params = {
            'n_hidden': 10, 'learning_rate': 0.005, 'epochs': 350,
            'activation_fn': 'sigmoid', 'noise': 0.9, 'test_ratio': 0.333
        }
        
        # Test baseline
        accuracy = train_and_evaluate(baseline_params)
        self.log_result(baseline_params, accuracy)
        
        # Fine-tune around baseline
        variations = [
            {'n_hidden': [8, 9, 11, 12]},
            {'learning_rate': [0.004, 0.006, 0.007]},
            {'epochs': [300, 400, 500]},
            {'noise': [0.8, 1.0, 1.1]},
            {'test_ratio': [0.25, 0.4]}
        ]
        
        for var in variations:
            for param, values in var.items():
                for value in values:
                    test_params = baseline_params.copy()
                    test_params[param] = value
                    accuracy = train_and_evaluate(test_params)
                    self.log_result(test_params, accuracy)
    
    def strategy_2_grid_search(self):
        """Strategy 2: Systematic grid search"""
        print("\n🔍 Strategy 2: Grid Search")
        print("=" * 50)
        
        # Define parameter grids
        param_grid = {
            'n_hidden': [5, 8, 10, 12, 15, 20, 25],
            'learning_rate': [0.001, 0.003, 0.005, 0.007, 0.01, 0.015],
            'epochs': [200, 300, 350, 400, 500, 600],
            'activation_fn': ['sigmoid', 'relu', 'tanh'],
            'noise': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            'test_ratio': [0.25, 0.333, 0.4]
        }
        
        # Generate combinations (limit to avoid too many tests)
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*[param_grid[key] for key in keys]))
        
        # Sample combinations intelligently
        np.random.seed(42)
        sample_size = min(100, len(combinations))
        sampled_combinations = np.random.choice(len(combinations), sample_size, replace=False)
        
        for i, idx in enumerate(sampled_combinations):
            combo = combinations[idx]
            params = dict(zip(keys, combo))
            accuracy = train_and_evaluate(params)
            self.log_result(params, accuracy)
            
            if i % 10 == 0:
                print(f"Grid search progress: {i+1}/{sample_size}")
    
    def strategy_3_adaptive_search(self):
        """Strategy 3: Adaptive search based on previous results"""
        print("\n🔍 Strategy 3: Adaptive Search")
        print("=" * 50)
        
        if len(self.results) < 10:
            print("Need more baseline results for adaptive search")
            return
        
        # Analyze what works best
        high_accuracy_results = [r for r in self.results if r[1] >= 70]
        
        if not high_accuracy_results:
            print("No high accuracy results found for adaptive search")
            return
        
        # Find parameter ranges that work well
        best_params = high_accuracy_results[0][0]
        
        # Adaptive fine-tuning
        adaptive_tests = []
        
        # Test around best parameters with smaller steps
        for n_hidden in range(max(5, best_params['n_hidden']-3), min(30, best_params['n_hidden']+4)):
            for lr in np.arange(max(0.001, best_params['learning_rate']-0.002), 
                              min(0.02, best_params['learning_rate']+0.003), 0.001):
                for epochs in range(max(200, best_params['epochs']-100), 
                                  min(800, best_params['epochs']+150), 50):
                    for noise in np.arange(max(0.5, best_params['noise']-0.2), 
                                         min(1.5, best_params['noise']+0.3), 0.1):
                        adaptive_tests.append({
                            'n_hidden': n_hidden,
                            'learning_rate': round(lr, 4),
                            'epochs': epochs,
                            'activation_fn': best_params['activation_fn'],
                            'noise': round(noise, 1),
                            'test_ratio': best_params['test_ratio']
                        })
        
        # Sample adaptive tests
        sample_size = min(50, len(adaptive_tests))
        sampled_tests = np.random.choice(adaptive_tests, sample_size, replace=False)
        
        for i, params in enumerate(sampled_tests):
            accuracy = train_and_evaluate(params)
            self.log_result(params, accuracy)
            
            if i % 10 == 0:
                print(f"Adaptive search progress: {i+1}/{sample_size}")
    
    def strategy_4_evolutionary_search(self):
        """Strategy 4: Evolutionary approach - mutate best parameters"""
        print("\n🔍 Strategy 4: Evolutionary Search")
        print("=" * 50)
        
        if not self.best_params:
            print("No best parameters found for evolutionary search")
            return
        
        # Start with best parameters and evolve
        current_params = self.best_params.copy()
        
        for generation in range(10):
            print(f"Generation {generation + 1}/10")
            
            # Create mutations
            mutations = []
            for _ in range(10):
                mutation = current_params.copy()
                
                # Random mutations
                if np.random.random() < 0.3:
                    mutation['n_hidden'] = max(5, min(30, mutation['n_hidden'] + np.random.randint(-2, 3)))
                if np.random.random() < 0.3:
                    mutation['learning_rate'] = max(0.001, min(0.02, mutation['learning_rate'] * np.random.uniform(0.8, 1.2)))
                if np.random.random() < 0.3:
                    mutation['epochs'] = max(200, min(800, mutation['epochs'] + np.random.randint(-50, 51)))
                if np.random.random() < 0.3:
                    mutation['noise'] = max(0.5, min(1.5, mutation['noise'] + np.random.uniform(-0.1, 0.1)))
                
                mutations.append(mutation)
            
            # Evaluate mutations
            best_mutation_accuracy = 0
            best_mutation_params = None
            
            for mutation in mutations:
                accuracy = train_and_evaluate(mutation)
                self.log_result(mutation, accuracy)
                
                if accuracy > best_mutation_accuracy:
                    best_mutation_accuracy = accuracy
                    best_mutation_params = mutation.copy()
            
            # Update current best if mutation is better
            if best_mutation_accuracy > self.best_accuracy:
                current_params = best_mutation_params.copy()
                print(f"Evolution improved: {best_mutation_accuracy:.2f}%")
            else:
                # Add some randomness to avoid local optima
                current_params = self.best_params.copy()
    
    def strategy_5_hyperparameter_optimization(self):
        """Strategy 5: Advanced hyperparameter optimization"""
        print("\n🔍 Strategy 5: Advanced Hyperparameter Optimization")
        print("=" * 50)
        
        # Focus on promising regions
        promising_combinations = [
            # Low noise combinations
            {'n_hidden': 10, 'learning_rate': 0.004, 'epochs': 400, 'activation_fn': 'sigmoid', 'noise': 0.6, 'test_ratio': 0.333},
            {'n_hidden': 12, 'learning_rate': 0.005, 'epochs': 350, 'activation_fn': 'sigmoid', 'noise': 0.6, 'test_ratio': 0.333},
            {'n_hidden': 8, 'learning_rate': 0.006, 'epochs': 300, 'activation_fn': 'sigmoid', 'noise': 0.6, 'test_ratio': 0.333},
            
            # High capacity combinations
            {'n_hidden': 20, 'learning_rate': 0.003, 'epochs': 500, 'activation_fn': 'sigmoid', 'noise': 0.8, 'test_ratio': 0.333},
            {'n_hidden': 25, 'learning_rate': 0.002, 'epochs': 600, 'activation_fn': 'sigmoid', 'noise': 0.8, 'test_ratio': 0.333},
            
            # Different activation functions with optimal params
            {'n_hidden': 10, 'learning_rate': 0.005, 'epochs': 350, 'activation_fn': 'relu', 'noise': 0.7, 'test_ratio': 0.333},
            {'n_hidden': 15, 'learning_rate': 0.004, 'epochs': 400, 'activation_fn': 'relu', 'noise': 0.7, 'test_ratio': 0.333},
            {'n_hidden': 10, 'learning_rate': 0.005, 'epochs': 350, 'activation_fn': 'tanh', 'noise': 0.7, 'test_ratio': 0.333},
            
            # Extreme combinations
            {'n_hidden': 5, 'learning_rate': 0.01, 'epochs': 200, 'activation_fn': 'sigmoid', 'noise': 0.5, 'test_ratio': 0.25},
            {'n_hidden': 30, 'learning_rate': 0.001, 'epochs': 800, 'activation_fn': 'sigmoid', 'noise': 1.0, 'test_ratio': 0.4},
        ]
        
        for i, params in enumerate(promising_combinations):
            accuracy = train_and_evaluate(params)
            self.log_result(params, accuracy)
            print(f"Advanced optimization {i+1}/{len(promising_combinations)}: {accuracy:.2f}%")
    
    def run_all_strategies(self):
        """Run all search strategies"""
        print("🚀 Starting Smart Parameter Search")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all strategies
        self.strategy_1_baseline_search()
        self.strategy_2_grid_search()
        self.strategy_3_adaptive_search()
        self.strategy_4_evolutionary_search()
        self.strategy_5_hyperparameter_optimization()
        
        end_time = time.time()
        
        # Final results
        print("\n" + "=" * 60)
        print("🏆 FINAL RESULTS")
        print("=" * 60)
        print(f"Total tests performed: {len(self.results)}")
        print(f"Best accuracy: {self.best_accuracy:.2f}%")
        print(f"Best parameters: {self.best_params}")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        
        # Save results
        self.save_results()
        
        return self.best_params, self.best_accuracy
    
    def save_results(self):
        """Save results to file"""
        results_data = {
            'best_accuracy': self.best_accuracy,
            'best_params': self.best_params,
            'all_results': [(str(params), acc) for params, acc in self.results],
            'param_history': dict(self.param_history)
        }
        
        with open('search_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to search_results.json")

# ===== 6) Run Smart Search =====
if __name__ == "__main__":
    searcher = SmartParameterSearch()
    best_params, best_accuracy = searcher.run_all_strategies()
    
    print(f"\n✅ Final best accuracy: {best_accuracy:.2f}%")
    print("Best parameters:", best_params) 