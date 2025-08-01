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

def fast_optimize():
    """Fast two-stage optimization"""
    
    print("🚀 Быстрая оптимизация гиперпараметров")
    print("=" * 50)
    
    # Stage 1: Coarse search (fewer combinations)
    print("📊 Этап 1: Грубый поиск...")
    
    # Coarse parameter ranges
    n_hidden_coarse = [20, 80, 150, 200]
    lr_coarse = [0.001, 0.005, 0.01, 0.05, 0.1]
    epochs_coarse = [500, 1500, 2500]
    activation_coarse = ['sigmoid', 'relu', 'tanh']
    noise_coarse = [0.5, 0.8, 1.0, 1.3, 1.5]
    test_ratio_coarse = [0.2, 0.3, 0.4, 0.5]
    
    best_coarse_acc = 0
    best_coarse_params = None
    coarse_results = []
    
    start_time = time.time()
    count = 0
    total_coarse = len(n_hidden_coarse) * len(lr_coarse) * len(epochs_coarse) * len(activation_coarse) * len(noise_coarse) * len(test_ratio_coarse)
    
    for n_hidden in n_hidden_coarse:
        for lr in lr_coarse:
            for epochs in epochs_coarse:
                for act in activation_coarse:
                    for noise in noise_coarse:
                        for test_ratio in test_ratio_coarse:
                            count += 1
                            
                            params = {
                                'n_hidden': n_hidden,
                                'learning_rate': lr,
                                'epochs': epochs,
                                'activation_fn': act,
                                'noise': noise,
                                'test_ratio': test_ratio
                            }
                            
                            try:
                                accuracy = train_and_evaluate(params)
                                coarse_results.append((accuracy, params))
                                
                                if accuracy > best_coarse_acc:
                                    best_coarse_acc = accuracy
                                    best_coarse_params = params.copy()
                                    print(f"🎯 Грубый поиск: {accuracy:.2f}% | {params}")
                                
                                if count % 50 == 0:
                                    print(f"📈 Прогресс: {count}/{total_coarse} ({count/total_coarse*100:.1f}%)")
                                
                            except Exception as e:
                                continue
    
    coarse_time = time.time() - start_time
    print(f"✅ Грубый поиск завершен за {coarse_time:.1f} сек")
    print(f"🏆 Лучший результат грубого поиска: {best_coarse_acc:.2f}%")
    
    # Stage 2: Fine search around best coarse result
    print("\n🔍 Этап 2: Точный поиск вокруг лучшего результата...")
    
    # Fine parameter ranges around best coarse result
    best = best_coarse_params
    
    # Define fine search ranges
    n_hidden_fine = [max(10, best['n_hidden']-20), best['n_hidden']-10, best['n_hidden'], 
                     best['n_hidden']+10, min(200, best['n_hidden']+20)]
    n_hidden_fine = list(set(n_hidden_fine))  # Remove duplicates
    
    lr_fine = [best['learning_rate']*0.5, best['learning_rate']*0.8, best['learning_rate'], 
               best['learning_rate']*1.2, best['learning_rate']*1.5]
    lr_fine = [max(0.001, min(0.1, lr)) for lr in lr_fine]  # Clamp to range
    
    epochs_fine = [max(100, best['epochs']-500), best['epochs']-250, best['epochs'], 
                   best['epochs']+250, min(3000, best['epochs']+500)]
    epochs_fine = list(set(epochs_fine))
    
    noise_fine = [max(0.5, best['noise']-0.2), best['noise']-0.1, best['noise'], 
                  best['noise']+0.1, min(1.5, best['noise']+0.2)]
    noise_fine = list(set(noise_fine))
    
    test_ratio_fine = [max(0.2, best['test_ratio']-0.05), best['test_ratio']-0.025, best['test_ratio'], 
                       best['test_ratio']+0.025, min(0.5, best['test_ratio']+0.05)]
    test_ratio_fine = list(set(test_ratio_fine))
    
    best_fine_acc = best_coarse_acc
    best_fine_params = best_coarse_params.copy()
    fine_results = []
    
    start_time = time.time()
    count = 0
    total_fine = len(n_hidden_fine) * len(lr_fine) * len(epochs_fine) * len(activation_coarse) * len(noise_fine) * len(test_ratio_fine)
    
    for n_hidden in n_hidden_fine:
        for lr in lr_fine:
            for epochs in epochs_fine:
                for act in activation_coarse:
                    for noise in noise_fine:
                        for test_ratio in test_ratio_fine:
                            count += 1
                            
                            params = {
                                'n_hidden': n_hidden,
                                'learning_rate': lr,
                                'epochs': epochs,
                                'activation_fn': act,
                                'noise': noise,
                                'test_ratio': test_ratio
                            }
                            
                            try:
                                accuracy = train_and_evaluate(params)
                                fine_results.append((accuracy, params))
                                
                                if accuracy > best_fine_acc:
                                    best_fine_acc = accuracy
                                    best_fine_params = params.copy()
                                    print(f"🎯 Точный поиск: {accuracy:.2f}% | {params}")
                                
                                if count % 20 == 0:
                                    print(f"📈 Прогресс: {count}/{total_fine} ({count/total_fine*100:.1f}%)")
                                
                            except Exception as e:
                                continue
    
    fine_time = time.time() - start_time
    total_time = coarse_time + fine_time
    
    print(f"\n✅ Оптимизация завершена за {total_time:.1f} сек")
    print(f"🏆 Финальная лучшая точность: {best_fine_acc:.2f}%")
    print(f"🎯 Оптимальные параметры:")
    for key, value in best_fine_params.items():
        print(f"   {key}: {value}")
    
    # Combine and sort all results
    all_results = coarse_results + fine_results
    all_results.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\n📈 Топ-5 результатов:")
    for i, (acc, params) in enumerate(all_results[:5]):
        print(f"{i+1}. {acc:.2f}% - {params}")
    
    return best_fine_params, best_fine_acc

if __name__ == "__main__":
    best_params, best_accuracy = fast_optimize()
    
    # Save best parameters to a file
    with open('best_params_fast.txt', 'w') as f:
        f.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
        f.write("Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"    '{key}': {value},\n")
    
    print(f"\n💾 Лучшие параметры сохранены в файл 'best_params_fast.txt'")
    
    # Also update the original aa.py file with best parameters
    print(f"\n🔄 Обновление файла aa.py с лучшими параметрами...")
    
    # Read the original file
    with open('aa.py', 'r') as f:
        content = f.read()
    
    # Replace the params section
    params_section = f"""params = {{
    'n_hidden'     : {best_params['n_hidden']},        # Hidden layer size  | Range: 10 ~ 200
    'learning_rate': {best_params['learning_rate']},      # Learning rate      | Range: 0.001 ~ 0.1 (log scale)
    'epochs'       : {best_params['epochs']},       # Training epochs    | Range: 100 ~ 3000
    'activation_fn': '{best_params['activation_fn']}', # Activation fn      | Options: 'sigmoid', 'relu', 'tanh'
    'noise'        : {best_params['noise']},       # Input noise        | Range: 0.5 ~ 1.5 (optional; for robustness test)
    'test_ratio'   : {best_params['test_ratio']}        # Test split ratio   | Range: 0.2 ~ 0.5 (optional; affects evaluation scale)
}}"""
    
    # Find and replace the params section
    import re
    pattern = r'params = \{[\s\S]*?\}'
    new_content = re.sub(pattern, params_section, content)
    
    # Write back to file
    with open('aa.py', 'w') as f:
        f.write(new_content)
    
    print(f"✅ Файл aa.py обновлен с лучшими параметрами!")
    print(f"🎯 Теперь можно запустить: python aa.py") 