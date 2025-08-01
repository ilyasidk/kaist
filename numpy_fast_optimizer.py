import numpy as np
import time
from itertools import product
import multiprocessing as mp
import os

# Constants
N_POINTS = 150
N_INPUT = 2
N_OUTPUT = 1

def make_extreme_spiral(n_points, noise=1.0):
    np.random.seed(0)
    theta = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    r_a = 2*theta + np.pi
    r_b = -2*theta - np.pi

    x1 = r_a * np.cos(theta) + np.random.randn(n_points, 1)*noise
    y1 = r_a * np.sin(theta) + np.random.randn(n_points, 1)*noise
    x2 = r_b * np.cos(theta) + np.random.randn(n_points, 1)*noise
    y2 = r_b * np.sin(theta) + np.random.randn(n_points, 1)*noise

    X = np.vstack((np.hstack((x1,y1)), np.hstack((x2,y2))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def tanh_activation(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.tanh(z)**2

def train_and_evaluate_fast(params_tuple):
    """Ultra-fast training with NumPy vectorization"""
    n_hidden, learning_rate, epochs, activation_fn, noise, test_ratio = params_tuple
    
    try:
        # Generate dataset
        X, y = make_extreme_spiral(N_POINTS, noise)
        
        # Split data
        n_test = int(len(X) * test_ratio)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]
        
        # Select activation function
        if activation_fn == 'sigmoid':
            activation = sigmoid
            activation_prime = sigmoid_prime
        elif activation_fn == 'relu':
            activation = relu
            activation_prime = relu_prime
        elif activation_fn == 'tanh':
            activation = tanh_activation
            activation_prime = tanh_prime
        else:
            return 0.0
        
        # Initialize weights
        np.random.seed(0)
        W1 = np.random.randn(N_INPUT, n_hidden) * 0.1
        b1 = np.zeros((1, n_hidden))
        W2 = np.random.randn(n_hidden, N_OUTPUT) * 0.1
        b2 = np.zeros((1, N_OUTPUT))
        
        # Vectorized training
        for ep in range(epochs):
            # Forward pass
            Z1 = X_train.dot(W1) + b1
            A1 = activation(Z1)
            Z2 = A1.dot(W2) + b2
            A2 = sigmoid(Z2)
            
            # Backward pass
            dZ2 = (A2.reshape(-1, 1) - y_train.reshape(-1, 1)) * sigmoid_prime(Z2)
            dW2 = A1.T.dot(dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            dA1 = dZ2.dot(W2.T)
            dZ1 = dA1 * activation_prime(Z1)
            dW1 = X_train.T.dot(dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            # Update weights
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
        
        # Prediction
        Z1_test = X_test.dot(W1) + b1
        A1_test = activation(Z1_test)
        Z2_test = A1_test.dot(W2) + b2
        A2_test = sigmoid(Z2_test)
        predictions = (A2_test > 0.5).astype(int).flatten()
        
        accuracy = (predictions == y_test).mean() * 100
        return accuracy
        
    except:
        return 0.0

def numpy_ultra_fast_optimize():
    """Ultra-fast optimization with NumPy"""
    
    print("⚡ NUMPY ULTRA-FAST OPTIMIZATION")
    print("=" * 50)
    
    # Smart parameter selection
    n_hidden_values = [20, 40, 60, 80, 100, 120, 150, 180, 200]
    lr_values = [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
    epochs_values = [200, 400, 600, 800, 1000, 1200, 1500, 2000]
    activation_fns = ['sigmoid', 'tanh']
    noise_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    test_ratio_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    
    # Generate combinations
    all_combinations = list(product(
        n_hidden_values, lr_values, epochs_values, 
        activation_fns, noise_values, test_ratio_values
    ))
    
    total_combinations = len(all_combinations)
    print(f"🔍 Combinations: {total_combinations:,}")
    print(f"🚀 Starting parallel processing...")
    
    # Use all CPU cores
    num_cores = mp.cpu_count()
    print(f"💻 Using {num_cores} CPU cores")
    
    start_time = time.time()
    
    # Parallel processing
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(train_and_evaluate_fast, all_combinations)
    
    # Find best result
    best_idx = np.argmax(results)
    best_accuracy = results[best_idx]
    best_params = all_combinations[best_idx]
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ OPTIMIZATION COMPLETED!")
    print(f"⏱️  Time: {elapsed_time:.1f} seconds")
    print(f"📊 Speed: {total_combinations/elapsed_time:.0f} combinations/sec")
    print(f"🏆 Best accuracy: {best_accuracy:.2f}%")
    
    # Save results
    with open('numpy_best_params.txt', 'w') as f:
        f.write(f"NumPy Ultra-Fast Optimization Results\n")
        f.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Time: {elapsed_time:.1f} seconds\n")
        f.write(f"Combinations: {total_combinations:,}\n\n")
        f.write("Best Parameters:\n")
        f.write(f"    'n_hidden': {best_params[0]},\n")
        f.write(f"    'learning_rate': {best_params[1]},\n")
        f.write(f"    'epochs': {best_params[2]},\n")
        f.write(f"    'activation_fn': '{best_params[3]}',\n")
        f.write(f"    'noise': {best_params[4]},\n")
        f.write(f"    'test_ratio': {best_params[5]}\n")
    
    # Update aa.py file
    print(f"\n🔄 Updating aa.py...")
    
    with open('aa.py', 'r') as f:
        content = f.read()
    
    params_section = f"""params = {{
    'n_hidden'     : {best_params[0]},        # Hidden layer size  | Range: 10 ~ 200
    'learning_rate': {best_params[1]},      # Learning rate      | Range: 0.001 ~ 0.1 (log scale)
    'epochs'       : {best_params[2]},       # Training epochs    | Range: 100 ~ 3000
    'activation_fn': '{best_params[3]}', # Activation fn      | Options: 'sigmoid', 'relu', 'tanh'
    'noise'        : {best_params[4]},       # Input noise        | Range: 0.5 ~ 1.5 (optional; for robustness test)
    'test_ratio'   : {best_params[5]}        # Test split ratio   | Range: 0.2 ~ 0.5 (optional; affects evaluation scale)
}}"""
    
    import re
    pattern = r'params = \{[\s\S]*?\}'
    new_content = re.sub(pattern, params_section, content)
    
    with open('aa.py', 'w') as f:
        f.write(new_content)
    
    print(f"✅ File aa.py updated!")
    print(f"🎯 Run: python aa.py")
    
    return best_params, best_accuracy

if __name__ == "__main__":
    # Set multiprocessing start method for Windows
    if os.name == 'nt':
        mp.set_start_method('spawn', force=True)
    
    best_params, best_accuracy = numpy_ultra_fast_optimize() 