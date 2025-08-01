import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import multiprocessing as mp
from itertools import product
import os

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

def train_and_evaluate_single(params_tuple):
    """Train neural network with given parameters and return accuracy"""
    n_hidden, learning_rate, epochs, activation_fn, noise, test_ratio = params_tuple
    
    params = {
        'n_hidden': n_hidden,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'activation_fn': activation_fn,
        'noise': noise,
        'test_ratio': test_ratio
    }
    
    try:
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
            return 0.0, params
        
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
        return accuracy, params
        
    except Exception as e:
        return 0.0, params

def super_quick_optimize():
    """Super quick optimization with smart parameter selection"""
    
    print("🚀 СУПЕР-БЫСТРАЯ ОПТИМИЗАЦИЯ")
    print("=" * 50)
    
    # SMART PARAMETER SELECTION - most promising combinations
    # Based on neural network theory and spiral dataset characteristics
    
    # For spiral data: larger networks, moderate learning rates, sigmoid/tanh work better
    n_hidden_values = [15, 25, 40, 60, 80, 100, 120, 150, 180, 200]  # Focus on medium-large
    learning_rate_values = [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]  # Moderate rates
    epochs_values = [300, 500, 800, 1200, 1500, 2000, 2500]  # Sufficient training
    activation_fns = ['sigmoid', 'tanh']  # Better for spiral data than ReLU
    noise_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # Lower noise often better
    test_ratio_values = [0.2, 0.25, 0.3, 0.35, 0.4]  # Standard splits
    
    # Generate combinations
    all_combinations = list(product(
        n_hidden_values, learning_rate_values, epochs_values, 
        activation_fns, noise_values, test_ratio_values
    ))
    
    total_combinations = len(all_combinations)
    print(f"🔍 Комбинаций для тестирования: {total_combinations:,}")
    print(f"🚀 Запуск параллельной обработки...")
    
    # Use all CPU cores
    num_cores = mp.cpu_count()
    print(f"💻 Используем {num_cores} ядер процессора")
    
    start_time = time.time()
    
    # Parallel processing
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(train_and_evaluate_single, all_combinations)
    
    # Process results
    valid_results = [(acc, params) for acc, params in results if acc > 0]
    valid_results.sort(key=lambda x: x[0], reverse=True)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    print(f"⏱️  Время выполнения: {elapsed_time:.1f} сек")
    print(f"📊 Скорость: {total_combinations/elapsed_time:.0f} комбинаций/сек")
    
    if valid_results:
        best_accuracy, best_params = valid_results[0]
        print(f"🏆 ЛУЧШАЯ ТОЧНОСТЬ: {best_accuracy:.2f}%")
        print(f"🎯 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        
        # Show top 10 results
        print(f"\n📈 ТОП-10 РЕЗУЛЬТАТОВ:")
        for i, (acc, params) in enumerate(valid_results[:10]):
            print(f"{i+1:2d}. {acc:.2f}% - {params}")
        
        # Save results
        with open('super_quick_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"Super Quick Optimization Results\n")
            f.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
            f.write(f"Time: {elapsed_time:.1f} seconds\n")
            f.write(f"Combinations tested: {total_combinations:,}\n\n")
            f.write("Best Parameters:\n")
            for key, value in best_params.items():
                f.write(f"    '{key}': {value},\n")
            
            f.write(f"\nTop 10 Results:\n")
            for i, (acc, params) in enumerate(valid_results[:10]):
                f.write(f"{i+1}. {acc:.2f}% - {params}\n")
        
        # Update original aa.py file
        print(f"\n🔄 Обновление файла aa.py...")
        
        with open('aa.py', 'r', encoding='utf-8') as f:
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
        
        with open('aa.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Файл aa.py обновлен!")
        print(f"🎯 Запустите: python aa.py")
        
        return best_params, best_accuracy
    else:
        print("❌ Не найдено валидных результатов")
        return None, 0

if __name__ == "__main__":
    # Set multiprocessing start method for Windows
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    
    best_params, best_accuracy = super_quick_optimize() 