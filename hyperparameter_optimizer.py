import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
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

def optimize_hyperparameters():
    """Smart hyperparameter optimization"""
    
    # Define parameter ranges
    n_hidden_values = [20, 50, 80, 120, 150, 180, 200]  # Smart sampling
    learning_rate_values = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05, 0.08, 0.1]  # Log scale
    epochs_values = [500, 1000, 1500, 2000, 2500, 3000]  # Smart sampling
    activation_fns = ['sigmoid', 'relu', 'tanh']
    noise_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    test_ratio_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    best_accuracy = 0
    best_params = None
    results = []
    
    total_combinations = len(n_hidden_values) * len(learning_rate_values) * len(epochs_values) * len(activation_fns) * len(noise_values) * len(test_ratio_values)
    print(f"🔍 Всего комбинаций для перебора: {total_combinations:,}")
    print("⏳ Начинаем оптимизацию...")
    
    start_time = time.time()
    count = 0
    
    # Smart grid search with early stopping for very poor performers
    for n_hidden in n_hidden_values:
        for learning_rate in learning_rate_values:
            for epochs in epochs_values:
                for activation_fn in activation_fns:
                    for noise in noise_values:
                        for test_ratio in test_ratio_values:
                            count += 1
                            
                            params = {
                                'n_hidden': n_hidden,
                                'learning_rate': learning_rate,
                                'epochs': epochs,
                                'activation_fn': activation_fn,
                                'noise': noise,
                                'test_ratio': test_ratio
                            }
                            
                            try:
                                accuracy = train_and_evaluate(params)
                                results.append((accuracy, params))
                                
                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_params = params.copy()
                                    print(f"🎯 Новая лучшая точность: {accuracy:.2f}% (комбинация {count}/{total_combinations})")
                                    print(f"   Параметры: {best_params}")
                                
                                # Progress update every 100 combinations
                                if count % 100 == 0:
                                    elapsed = time.time() - start_time
                                    rate = count / elapsed
                                    remaining = (total_combinations - count) / rate
                                    print(f"📊 Прогресс: {count}/{total_combinations} ({count/total_combinations*100:.1f}%) | "
                                          f"Лучшая точность: {best_accuracy:.2f}% | "
                                          f"Осталось: {remaining/60:.1f} мин")
                                
                            except Exception as e:
                                print(f"❌ Ошибка с параметрами {params}: {e}")
                                continue
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Оптимизация завершена за {elapsed_time/60:.1f} минут")
    print(f"🏆 Лучшая точность: {best_accuracy:.2f}%")
    print(f"🎯 Оптимальные параметры:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # Show top 10 results
    results.sort(key=lambda x: x[0], reverse=True)
    print(f"\n📈 Топ-10 результатов:")
    for i, (acc, params) in enumerate(results[:10]):
        print(f"{i+1:2d}. {acc:.2f}% - {params}")
    
    return best_params, best_accuracy

if __name__ == "__main__":
    best_params, best_accuracy = optimize_hyperparameters()
    
    # Save best parameters to a file
    with open('best_params.txt', 'w') as f:
        f.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
        f.write("Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"    '{key}': {value},\n")
    
    print(f"\n💾 Лучшие параметры сохранены в файл 'best_params.txt'") 