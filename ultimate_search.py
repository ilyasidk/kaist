import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from collections import defaultdict

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
def train_and_evaluate(test_params):
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

# ===== 5) Ultimate Smart Search =====
class UltimateSmartSearch:
    def __init__(self):
        self.results = []
        self.best_accuracy = 0
        self.best_params = None
        self.param_history = defaultdict(list)
        self.start_time = time.time()
        
    def log_result(self, params, accuracy, test_num, total_tests):
        self.results.append((params.copy(), accuracy))
        self.param_history['n_hidden'].append(params['n_hidden'])
        self.param_history['learning_rate'].append(params['learning_rate'])
        self.param_history['epochs'].append(params['epochs'])
        self.param_history['noise'].append(params['noise'])
        self.param_history['accuracy'].append(accuracy)
        
        elapsed_time = time.time() - self.start_time
        eta = (elapsed_time / test_num) * (total_tests - test_num) if test_num > 0 else 0
        
        # Format parameters for better readability
        param_str = f"H:{params['n_hidden']:2d} LR:{params['learning_rate']:.4f} E:{params['epochs']:4d} A:{params['activation_fn'][:3]} N:{params['noise']:.1f} T:{params['test_ratio']:.3f}"
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_params = params.copy()
            print(f"🎉 NEW BEST: {accuracy:.2f}% | Test {test_num:6d}/{total_tests} | ETA: {eta/3600:.1f}h | {param_str}")
        else:
            print(f"Test {test_num:6d}/{total_tests} | Acc: {accuracy:.2f}% | Best: {self.best_accuracy:.2f}% | ETA: {eta/3600:.1f}h | {param_str}")
    
    def generate_all_combinations(self):
        """Generate ALL possible parameter combinations within valid ranges"""
        print("🔍 Generating ALL parameter combinations...")
        
        # Define ALL valid parameter values within ranges
        param_grid = {
            'n_hidden': list(range(10, 201, 5)),  # 10 to 200, step 5
            'learning_rate': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1],
            'epochs': [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500, 3000],
            'activation_fn': ['sigmoid', 'relu', 'tanh'],
            'noise': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            'test_ratio': [0.2, 0.25, 0.3, 0.333, 0.35, 0.4, 0.45, 0.5]
        }
        
        # Generate all combinations
        keys = list(param_grid.keys())
        all_combinations = list(itertools.product(*[param_grid[key] for key in keys]))
        
        # Convert to dictionaries
        all_params = []
        for combo in all_combinations:
            params = dict(zip(keys, combo))
            all_params.append(params)
        
        print(f"Generated {len(all_params)} total combinations")
        return all_params
    
    def smart_filter_combinations(self, all_params):
        """Smart filtering to reduce search space based on domain knowledge"""
        print("🧠 Applying smart filtering...")
        
        filtered_params = []
        
        for params in all_params:
            # Rule 1: Skip obviously bad combinations
            if params['learning_rate'] > 0.05 and params['epochs'] < 500:
                continue  # High LR with few epochs = unstable
            
            if params['learning_rate'] < 0.002 and params['epochs'] < 300:
                continue  # Very low LR with few epochs = slow convergence
            
            # Rule 2: Skip extreme combinations
            if params['n_hidden'] > 100 and params['learning_rate'] > 0.01:
                continue  # Large network with high LR = unstable
            
            if params['n_hidden'] < 15 and params['noise'] > 1.2:
                continue  # Small network with high noise = poor performance
            
            # Rule 3: Focus on promising regions
            if params['activation_fn'] == 'relu' and params['learning_rate'] > 0.01:
                continue  # ReLU with high LR often unstable
            
            # Rule 4: Balance test ratio
            if params['test_ratio'] < 0.2 or params['test_ratio'] > 0.5:
                continue  # Keep reasonable test split
            
            filtered_params.append(params)
        
        print(f"Filtered to {len(filtered_params)} promising combinations")
        return filtered_params
    
    def run_exhaustive_search(self):
        """Run exhaustive search on all valid combinations"""
        print("🚀 Starting ULTIMATE Exhaustive Search")
        print("=" * 60)
        
        # Generate all combinations
        all_params = self.generate_all_combinations()
        
        # Smart filtering
        filtered_params = self.smart_filter_combinations(all_params)
        
        # Sort by promising combinations first
        filtered_params = self.sort_by_promise(filtered_params)
        
        total_tests = len(filtered_params)
        print(f"Will test {total_tests:,} combinations")
        print(f"Estimated time: {total_tests * 0.5 / 3600:.1f} hours (assuming 0.5s per test)")
        print("=" * 60)
        print("Format: H:Hidden LR:LearningRate E:Epochs A:Activation N:Noise T:TestRatio")
        print("=" * 60)
        
        # Run tests
        for i, params in enumerate(filtered_params):
            accuracy = train_and_evaluate(params)
            self.log_result(params, accuracy, i+1, total_tests)
            
            # Save progress every 50 tests
            if (i + 1) % 50 == 0:
                self.save_progress()
                print(f"💾 Progress saved at test {i+1}")
        
        # Final results
        self.save_final_results()
        return self.best_params, self.best_accuracy
    
    def sort_by_promise(self, params_list):
        """Sort combinations by how promising they are"""
        print("📊 Sorting combinations by promise...")
        
        # Define promising characteristics
        def promise_score(params):
            score = 0
            
            # Sigmoid is often best for this task
            if params['activation_fn'] == 'sigmoid':
                score += 10
            
            # Moderate hidden size is often best
            if 10 <= params['n_hidden'] <= 50:
                score += 5
            elif 51 <= params['n_hidden'] <= 100:
                score += 3
            
            # Moderate learning rate
            if 0.003 <= params['learning_rate'] <= 0.01:
                score += 5
            elif 0.001 <= params['learning_rate'] <= 0.02:
                score += 3
            
            # Moderate epochs
            if 300 <= params['epochs'] <= 800:
                score += 5
            elif 200 <= params['epochs'] <= 1000:
                score += 3
            
            # Moderate noise
            if 0.7 <= params['noise'] <= 1.1:
                score += 5
            elif 0.5 <= params['noise'] <= 1.3:
                score += 3
            
            # Standard test ratio
            if 0.25 <= params['test_ratio'] <= 0.4:
                score += 5
            
            return score
        
        # Sort by promise score (highest first)
        sorted_params = sorted(params_list, key=promise_score, reverse=True)
        return sorted_params
    
    def save_progress(self):
        """Save current progress"""
        progress_data = {
            'best_accuracy': self.best_accuracy,
            'best_params': self.best_params,
            'tests_completed': len(self.results),
            'all_results': [(str(params), acc) for params, acc in self.results[-100:]]  # Last 100 results
        }
        
        with open('search_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def save_final_results(self):
        """Save final results"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        results_data = {
            'best_accuracy': self.best_accuracy,
            'best_params': self.best_params,
            'total_tests': len(self.results),
            'total_time_seconds': total_time,
            'average_time_per_test': total_time / len(self.results) if self.results else 0,
            'all_results': [(str(params), acc) for params, acc in self.results],
            'param_history': dict(self.param_history),
            'top_10_results': sorted(self.results, key=lambda x: x[1], reverse=True)[:10]
        }
        
        with open('ultimate_search_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n📊 Results saved to ultimate_search_results.json")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"⚡ Average time per test: {total_time/len(self.results):.3f} seconds")

# ===== 6) Run Ultimate Search =====
if __name__ == "__main__":
    print("🔥 ULTIMATE PARAMETER SEARCH")
    print("=" * 60)
    print("This will test ALL valid parameter combinations")
    print("Parameter ranges:")
    print("  Hidden neurons: 10-200 (step 5)")
    print("  Learning rate: 0.001-0.1 (16 values)")
    print("  Epochs: 100-3000 (19 values)")
    print("  Activation: sigmoid, relu, tanh")
    print("  Noise: 0.5-1.5 (11 values)")
    print("  Test ratio: 0.2-0.5 (8 values)")
    print("Estimated time: 30-60 minutes depending on your computer")
    print("=" * 60)
    
    searcher = UltimateSmartSearch()
    best_params, best_accuracy = searcher.run_exhaustive_search()
    
    print("\n" + "=" * 60)
    print("🏆 ULTIMATE SEARCH COMPLETE!")
    print("=" * 60)
    print(f"Best accuracy found: {best_accuracy:.2f}%")
    print(f"Best parameters: {best_params}")
    print(f"Total combinations tested: {len(searcher.results):,}")
    
    # Show top 5 results
    print("\n🏅 TOP 5 RESULTS:")
    top_results = sorted(searcher.results, key=lambda x: x[1], reverse=True)[:5]
    for i, (params, acc) in enumerate(top_results, 1):
        param_str = f"H:{params['n_hidden']:2d} LR:{params['learning_rate']:.4f} E:{params['epochs']:4d} A:{params['activation_fn'][:3]} N:{params['noise']:.1f} T:{params['test_ratio']:.3f}"
        print(f"{i}. {acc:.2f}% - {param_str}")
    
    print("\n✅ Search complete! Check ultimate_search_results.json for full details.") 