import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product
import json
import time
from datetime import datetime

# ===== Dataset =====
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

# ===== Activation Functions =====
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z): 
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 - s)

def relu(z): return np.maximum(0, z)
def relu_prime(z): return (z > 0).astype(np.float64)

def tanh(z): return np.tanh(z)
def tanh_prime(z): return 1.0 - np.tanh(z)**2

ACT_FUNCS = {
    "sigmoid": (sigmoid, sigmoid_prime),
    "relu": (relu, relu_prime),
    "tanh": (tanh, tanh_prime)
}

# ===== Training Function =====
def train_and_eval(params):
    n_hidden, lr, epochs, act_name, noise, test_ratio = params
    N_POINTS, N_INPUT, N_OUTPUT = 150, 2, 1

    X, y = make_extreme_spiral(N_POINTS, noise=noise)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=0
    )

    act, act_prime = ACT_FUNCS[act_name]

    np.random.seed(0)
    W1 = np.random.randn(N_INPUT, n_hidden) * 0.1
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, N_OUTPUT) * 0.1
    b2 = np.zeros((1, N_OUTPUT))

    for ep in range(epochs):
        Z1 = X_train.dot(W1) + b1
        A1 = act(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = sigmoid(Z2)

        dZ2 = (A2.reshape(-1,1) - y_train.reshape(-1,1)) * sigmoid_prime(Z2)
        dW2 = A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(W2.T)
        dZ1 = dA1 * act_prime(Z1)
        dW1 = X_train.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    # Prediction
    A1 = act(X_test.dot(W1) + b1)
    A2 = sigmoid(A1.dot(W2) + b2)
    y_pred = (A2 > 0.5).astype(int)
    acc = (y_pred.flatten() == y_test).mean() * 100
    return acc

# ===== GridSearch =====
grid = {
    'n_hidden': [10, 50, 100],
    'learning_rate': [0.001, 0.005, 0.01],
    'epochs': [200, 500],
    'activation_fn': ['sigmoid', 'relu', 'tanh'],
    'noise': [0.5, 1.0],
    'test_ratio': [0.2, 0.3]
}

# Параметры для быстрого тестирования
quick_grid = {
    'n_hidden': [10, 50],
    'learning_rate': [0.005, 0.01],
    'epochs': [200],
    'activation_fn': ['sigmoid', 'relu'],
    'noise': [0.5],
    'test_ratio': [0.2]
}

def run_grid_search(grid_params, save_results=True):
    best_score, best_params = -1, None
    results = []
    
    total_combinations = len(list(product(*grid_params.values())))
    print(f"🚀 Starting Grid Search with {total_combinations} combinations...")
    print(f"📊 Grid parameters: {grid_params}")
    
    start_time = time.time()
    
    for i, combo in enumerate(product(*grid_params.values())):
        acc = train_and_eval(combo)
        results.append({
            'params': combo,
            'accuracy': float(acc),
            'n_hidden': combo[0],
            'learning_rate': combo[1],
            'epochs': combo[2],
            'activation_fn': combo[3],
            'noise': combo[4],
            'test_ratio': combo[5]
        })
        
        if acc > best_score:
            best_score, best_params = acc, combo
            
        # Прогресс каждые 10 итераций
        if (i + 1) % 10 == 0 or i == total_combinations - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total_combinations - i - 1) if i > 0 else 0
            print(f"Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%) - "
                  f"Best: {best_score:.2f}% - ETA: {eta/60:.1f}min")
    
    total_time = time.time() - start_time
    
    # Сортируем результаты
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n✅ Search completed in {total_time/60:.1f} minutes")
    print(f"🎯 Best Accuracy: {best_score:.2f}%")
    print(f"🔧 Best Parameters: {best_params}")
    
    print(f"\n🔥 TOP-5 Results:")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"{i+1}. Acc={r['accuracy']:.2f}% - "
              f"Hidden={r['n_hidden']}, LR={r['learning_rate']}, "
              f"Epochs={r['epochs']}, Act={r['activation_fn']}, "
              f"Noise={r['noise']}, TestRatio={r['test_ratio']}")
    
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grid_search_results_{timestamp}.json"
        
        output_data = {
            'timestamp': timestamp,
            'total_time_minutes': total_time / 60,
            'total_combinations': total_combinations,
            'grid_parameters': grid_params,
            'best_accuracy': float(best_score),
            'best_params': best_params,
            'all_results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    return results, best_score, best_params

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 NEURAL NETWORK HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Сначала быстрый тест
    print("\n🔬 Running quick test first...")
    quick_results, quick_best_score, quick_best_params = run_grid_search(quick_grid, save_results=False)
    
    print(f"\n" + "=" * 60)
    print("🚀 Running full grid search...")
    print("=" * 60)
    
    # Полный поиск
    full_results, full_best_score, full_best_params = run_grid_search(grid, save_results=True)
    
    print(f"\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Quick test best: {quick_best_score:.2f}%")
    print(f"Full search best: {full_best_score:.2f}%")
    print(f"Improvement: {full_best_score - quick_best_score:.2f}%") 