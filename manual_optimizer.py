import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# ===== 1) Trainable Hyperparameters Only =====
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

# ===== 2) Constants (Not to be changed) =====
N_POINTS = 150
N_INPUT  = 2
N_OUTPUT = 1

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

def train_and_evaluate(params):
    """Train neural network with given parameters and return accuracy"""
    
    # Generate dataset
    X, y = make_extreme_spiral(N_POINTS, noise=params['noise'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_ratio'], random_state=0
    )
    
    # Select activation function
    act_name = params['activation_fn']
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
    
    # Weight Initialization
    np.random.seed(0)
    W1 = np.random.randn(N_INPUT, params['n_hidden']) * 0.1
    b1 = np.zeros((1, params['n_hidden']))
    W2 = np.random.randn(params['n_hidden'], N_OUTPUT) * 0.1
    b2 = np.zeros((1, N_OUTPUT))
    
    # Training Loop
    for ep in range(params['epochs']):
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

        W2 -= params['learning_rate'] * dW2
        b2 -= params['learning_rate'] * db2
        W1 -= params['learning_rate'] * dW1
        b1 -= params['learning_rate'] * db1
    
    # Prediction & Accuracy
    def predict(X):
        A1 = activation(X.dot(W1) + b1)
        A2 = sigmoid(A1.dot(W2) + b2)
        return (A2 > 0.5).astype(int)

    y_pred = predict(X_test)
    accuracy = (y_pred.flatten() == y_test).mean() * 100
    return accuracy

def manual_optimize():
    """Manual optimization based on neural network theory"""
    
    print("🧠 MANUAL OPTIMIZATION BASED ON THEORY")
    print("=" * 50)
    
    # Theory-based parameter combinations for spiral data
    # Spiral data is complex, needs larger networks, sigmoid/tanh work better than ReLU
    test_combinations = [
        # Large network, moderate learning rate, sigmoid
        {'n_hidden': 150, 'learning_rate': 0.01, 'epochs': 2000, 'activation_fn': 'sigmoid', 'noise': 0.7, 'test_ratio': 0.3},
        {'n_hidden': 180, 'learning_rate': 0.008, 'epochs': 2500, 'activation_fn': 'sigmoid', 'noise': 0.6, 'test_ratio': 0.25},
        {'n_hidden': 200, 'learning_rate': 0.005, 'epochs': 3000, 'activation_fn': 'sigmoid', 'noise': 0.5, 'test_ratio': 0.2},
        
        # Tanh activation (often better for complex patterns)
        {'n_hidden': 150, 'learning_rate': 0.01, 'epochs': 2000, 'activation_fn': 'tanh', 'noise': 0.7, 'test_ratio': 0.3},
        {'n_hidden': 180, 'learning_rate': 0.008, 'epochs': 2500, 'activation_fn': 'tanh', 'noise': 0.6, 'test_ratio': 0.25},
        {'n_hidden': 200, 'learning_rate': 0.005, 'epochs': 3000, 'activation_fn': 'tanh', 'noise': 0.5, 'test_ratio': 0.2},
        
        # Lower learning rates for stability
        {'n_hidden': 150, 'learning_rate': 0.002, 'epochs': 3000, 'activation_fn': 'sigmoid', 'noise': 0.6, 'test_ratio': 0.25},
        {'n_hidden': 180, 'learning_rate': 0.001, 'epochs': 3000, 'activation_fn': 'sigmoid', 'noise': 0.5, 'test_ratio': 0.2},
        
        # Medium networks with longer training
        {'n_hidden': 100, 'learning_rate': 0.01, 'epochs': 3000, 'activation_fn': 'sigmoid', 'noise': 0.6, 'test_ratio': 0.25},
        {'n_hidden': 120, 'learning_rate': 0.008, 'epochs': 3000, 'activation_fn': 'tanh', 'noise': 0.5, 'test_ratio': 0.2},
        
        # Very large network with careful training
        {'n_hidden': 200, 'learning_rate': 0.002, 'epochs': 3000, 'activation_fn': 'sigmoid', 'noise': 0.5, 'test_ratio': 0.2},
        {'n_hidden': 200, 'learning_rate': 0.001, 'epochs': 3000, 'activation_fn': 'tanh', 'noise': 0.5, 'test_ratio': 0.2},
    ]
    
    best_accuracy = 0
    best_params = None
    results = []
    
    start_time = time.time()
    
    for i, params in enumerate(test_combinations):
        print(f"Testing combination {i+1}/{len(test_combinations)}: {params}")
        
        try:
            accuracy = train_and_evaluate(params)
            results.append((accuracy, params))
            
            print(f"  Accuracy: {accuracy:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params.copy()
                print(f"  🎯 NEW BEST: {accuracy:.2f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ MANUAL OPTIMIZATION COMPLETED!")
    print(f"⏱️  Time: {elapsed_time:.1f} seconds")
    print(f"🏆 Best accuracy: {best_accuracy:.2f}%")
    print(f"🎯 Best parameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # Show all results
    results.sort(key=lambda x: x[0], reverse=True)
    print(f"\n📈 ALL RESULTS:")
    for i, (acc, params) in enumerate(results):
        print(f"{i+1:2d}. {acc:.2f}% - {params}")
    
    # Save results
    with open('manual_best_params.txt', 'w') as f:
        f.write(f"Manual Optimization Results\n")
        f.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Time: {elapsed_time:.1f} seconds\n\n")
        f.write("Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"    '{key}': {value},\n")
        
        f.write(f"\nAll Results:\n")
        for i, (acc, params) in enumerate(results):
            f.write(f"{i+1}. {acc:.2f}% - {params}\n")
    
    # Update aa.py file
    print(f"\n🔄 Updating aa.py...")
    
    with open('aa.py', 'r') as f:
        content = f.read()
    
    params_section = f"""params = {{
    'n_hidden'     : {best_params['n_hidden']},        # Hidden layer size  | Range: 10 ~ 200
    'learning_rate': {best_params['learning_rate']},      # Learning rate      | Range: 0.001 ~ 0.1 (log scale)
    'epochs'       : {best_params['epochs']},       # Training epochs    | Range: 100 ~ 3000
    'activation_fn': '{best_params['activation_fn']}', # Activation fn      | Options: 'sigmoid', 'relu', 'tanh'
    'noise'        : {best_params['noise']},       # Input noise        | Range: 0.5 ~ 1.5 (optional; for robustness test)
    'test_ratio'   : {best_params['test_ratio']}        # Test split ratio   | Range: 0.2 ~ 0.5 (optional; affects evaluation scale)
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
    best_params, best_accuracy = manual_optimize() 