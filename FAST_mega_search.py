import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import random

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

def leaky_relu(z, alpha=0.01): return np.where(z > 0, z, alpha * z)
def leaky_relu_prime(z, alpha=0.01): return np.where(z > 0, 1, alpha)

def elu(z, alpha=1.0): return np.where(z > 0, z, alpha * (np.exp(z) - 1))
def elu_prime(z, alpha=1.0): return np.where(z > 0, 1, alpha * np.exp(z))

ACT_FUNCS = {
    "sigmoid": (sigmoid, sigmoid_prime),
    "relu": (relu, relu_prime),
    "tanh": (tanh, tanh_prime),
    "leaky_relu": (leaky_relu, leaky_relu_prime),
    "elu": (elu, elu_prime)
}

# ===== Training Function =====
def train_and_eval(params):
    n_hidden, lr, epochs, act_name, noise, test_ratio, batch_size, momentum = params
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
    
    # Momentum
    vW1, vb1, vW2, vb2 = 0, 0, 0, 0

    for ep in range(epochs):
        # Mini-batch training
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            Z1 = X_batch.dot(W1) + b1
            A1 = act(Z1)
            Z2 = A1.dot(W2) + b2
            A2 = sigmoid(Z2)

            dZ2 = (A2.reshape(-1,1) - y_batch.reshape(-1,1)) * sigmoid_prime(Z2)
            dW2 = A1.T.dot(dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            dA1 = dZ2.dot(W2.T)
            dZ1 = dA1 * act_prime(Z1)
            dW1 = X_batch.T.dot(dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # Momentum update
            vW2 = momentum * vW2 + lr * dW2
            vb2 = momentum * vb2 + lr * db2
            vW1 = momentum * vW1 + lr * dW1
            vb1 = momentum * vb1 + lr * db1

            W2 -= vW2
            b2 -= vb2
            W1 -= vW1
            b1 -= vb1

    # Prediction
    A1 = act(X_test.dot(W1) + b1)
    A2 = sigmoid(A1.dot(W2) + b2)
    y_pred = (A2 > 0.5).astype(int)
    acc = (y_pred.flatten() == y_test).mean() * 100
    return acc

# ===== MEGA GridSearch =====
mega_grid = {
    'n_hidden': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
    'epochs': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000],
    'activation_fn': ['sigmoid', 'relu', 'tanh', 'leaky_relu', 'elu'],
    'noise': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0],
    'test_ratio': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    'batch_size': [1, 2, 4, 8, 16, 32, 64, 128],
    'momentum': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

# Подсчитываем общее количество комбинаций
total_combinations = 1
for values in mega_grid.values():
    total_combinations *= len(values)

print(f"🚀 MEGA GRID SEARCH")
print(f"📊 Total possible combinations: {total_combinations:,}")
print(f"⏰ Estimated time: {total_combinations * 0.1 / 60 / 60:.1f} hours")

def run_mega_search(max_combinations=1000000, save_results=True, create_plots=True):
    """Запускает мега-поиск с ограничением по количеству комбинаций"""
    
    best_score, best_params = -1, None
    results = []
    
    print(f"🎯 Running mega search with max {max_combinations:,} combinations...")
    print(f"📊 Grid parameters: {len(mega_grid)} parameters with {total_combinations:,} total combinations")
    
    start_time = time.time()
    
    # Генерируем случайные комбинации
    combinations_tested = 0
    
    while combinations_tested < max_combinations:
        # Случайная комбинация параметров
        combo = (
            random.choice(mega_grid['n_hidden']),
            random.choice(mega_grid['learning_rate']),
            random.choice(mega_grid['epochs']),
            random.choice(mega_grid['activation_fn']),
            random.choice(mega_grid['noise']),
            random.choice(mega_grid['test_ratio']),
            random.choice(mega_grid['batch_size']),
            random.choice(mega_grid['momentum'])
        )
        
        try:
            acc = train_and_eval(combo)
            results.append({
                'params': combo,
                'accuracy': float(acc),
                'n_hidden': combo[0],
                'learning_rate': combo[1],
                'epochs': combo[2],
                'activation_fn': combo[3],
                'noise': combo[4],
                'test_ratio': combo[5],
                'batch_size': combo[6],
                'momentum': combo[7]
            })
            
            if acc > best_score:
                best_score, best_params = acc, combo
                print(f"🔥 NEW BEST! Acc={acc:.2f}% - Params={combo}")
            
            combinations_tested += 1
            
            # Прогресс каждые 1000 итераций
            if combinations_tested % 1000 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / combinations_tested) * (max_combinations - combinations_tested) if combinations_tested > 0 else 0
                print(f"Progress: {combinations_tested:,}/{max_combinations:,} ({100*combinations_tested/max_combinations:.1f}%) - "
                      f"Best: {best_score:.2f}% - ETA: {eta/60:.1f}min")
                
                # Сохраняем промежуточные результаты
                if combinations_tested % 10000 == 0:
                    save_intermediate_results(results, combinations_tested)
                    
        except Exception as e:
            print(f"⚠️ Error with combo {combo}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Сортируем результаты
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n✅ Mega search completed in {total_time/60:.1f} minutes")
    print(f"🎯 Best Accuracy: {best_score:.2f}%")
    print(f"🔧 Best Parameters: {best_params}")
    print(f"📊 Combinations tested: {combinations_tested:,}")
    print(f"⚡ Speed: {combinations_tested/(total_time/60):.0f} combinations/minute")
    
    print(f"\n🔥 TOP-10 Results:")
    for i in range(min(10, len(results))):
        r = results[i]
        print(f"{i+1}. Acc={r['accuracy']:.2f}% - "
              f"Hidden={r['n_hidden']}, LR={r['learning_rate']}, "
              f"Epochs={r['epochs']}, Act={r['activation_fn']}, "
              f"Noise={r['noise']}, TestRatio={r['test_ratio']}, "
              f"Batch={r['batch_size']}, Momentum={r['momentum']}")
    
    if create_plots:
        create_mega_visualizations(results)
    
    if save_results:
        save_final_results(results, best_score, best_params, combinations_tested, total_time)
    
    return results, best_score, best_params

def save_intermediate_results(results, combinations_tested):
    """Сохраняет промежуточные результаты"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mega_search_intermediate_{combinations_tested}_{timestamp}.json"
    
    output_data = {
        'timestamp': timestamp,
        'combinations_tested': combinations_tested,
        'best_accuracy': max([r['accuracy'] for r in results]) if results else 0,
        'all_results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Intermediate results saved: {filename}")

def save_final_results(results, best_score, best_params, combinations_tested, total_time):
    """Сохраняет финальные результаты"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mega_search_final_{combinations_tested}_{timestamp}.json"
    
    output_data = {
        'timestamp': timestamp,
        'total_time_minutes': total_time / 60,
        'combinations_tested': combinations_tested,
        'total_possible_combinations': total_combinations,
        'coverage_percentage': (combinations_tested / total_combinations) * 100,
        'best_accuracy': float(best_score),
        'best_params': best_params,
        'all_results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Final results saved to: {filename}")

def create_mega_visualizations(results):
    """Создает визуализации для мега-поиска"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('MEGA Neural Network Hyperparameter Analysis', fontsize=16, fontweight='bold')
        
        # 1. Распределение точности
        accuracies = [r['accuracy'] for r in results]
        axes[0, 0].hist(accuracies, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Accuracies')
        axes[0, 0].set_xlabel('Accuracy (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.1f}%')
        axes[0, 0].legend()
        
        # 2. Точность по функциям активации
        act_acc = {}
        for r in results:
            act = r['activation_fn']
            if act not in act_acc:
                act_acc[act] = []
            act_acc[act].append(r['accuracy'])
        
        acts = list(act_acc.keys())
        acc_means = [np.mean(act_acc[act]) for act in acts]
        axes[0, 1].bar(acts, acc_means, color=['orange', 'green', 'purple', 'red', 'brown'])
        axes[0, 1].set_title('Average Accuracy by Activation Function')
        axes[0, 1].set_ylabel('Average Accuracy (%)')
        for i, v in enumerate(acc_means):
            axes[0, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
        # 3. Точность по learning rate
        lr_acc = {}
        for r in results:
            lr = r['learning_rate']
            if lr not in lr_acc:
                lr_acc[lr] = []
            lr_acc[lr].append(r['accuracy'])
        
        lrs = sorted(lr_acc.keys())
        lr_means = [np.mean(lr_acc[lr]) for lr in lrs]
        axes[0, 2].plot(lrs, lr_means, 'o-', color='lightgreen', linewidth=2, markersize=6)
        axes[0, 2].set_title('Average Accuracy by Learning Rate')
        axes[0, 2].set_xlabel('Learning Rate')
        axes[0, 2].set_ylabel('Average Accuracy (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Точность по количеству скрытых нейронов
        hidden_acc = {}
        for r in results:
            hidden = r['n_hidden']
            if hidden not in hidden_acc:
                hidden_acc[hidden] = []
            hidden_acc[hidden].append(r['accuracy'])
        
        hiddens = sorted(hidden_acc.keys())
        hidden_means = [np.mean(hidden_acc[h]) for h in hiddens]
        axes[1, 0].plot(hiddens, hidden_means, 'o-', color='lightcoral', linewidth=2, markersize=6)
        axes[1, 0].set_title('Average Accuracy by Hidden Units')
        axes[1, 0].set_xlabel('Hidden Units')
        axes[1, 0].set_ylabel('Average Accuracy (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Точность по momentum
        mom_acc = {}
        for r in results:
            mom = r['momentum']
            if mom not in mom_acc:
                mom_acc[mom] = []
            mom_acc[mom].append(r['accuracy'])
        
        moms = sorted(mom_acc.keys())
        mom_means = [np.mean(mom_acc[m]) for m in moms]
        axes[1, 1].plot(moms, mom_means, 'o-', color='lightblue', linewidth=2, markersize=6)
        axes[1, 1].set_title('Average Accuracy by Momentum')
        axes[1, 1].set_xlabel('Momentum')
        axes[1, 1].set_ylabel('Average Accuracy (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Топ-15 результатов
        top_15 = results[:15]
        top_acc = [r['accuracy'] for r in top_15]
        top_labels = [f"{r['n_hidden']}h,{r['learning_rate']}lr,{r['activation_fn']}" for r in top_15]
        
        axes[1, 2].barh(range(len(top_acc)), top_acc, color='gold')
        axes[1, 2].set_yticks(range(len(top_acc)))
        axes[1, 2].set_yticklabels(top_labels, fontsize=6)
        axes[1, 2].set_title('Top-15 Configurations')
        axes[1, 2].set_xlabel('Accuracy (%)')
        
        plt.tight_layout()
        
        # Сохраняем график
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"mega_hyperparameter_analysis_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Mega visualization saved to: {plot_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 MEGA NEURAL NETWORK HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"📊 Total possible combinations: {total_combinations:,}")
    print(f"🎯 This will test MILLIONS of combinations!")
    print("=" * 80)
    
    # Запускаем мега-поиск
    # Можно изменить количество комбинаций здесь
    max_combinations = 100000  # Начнем с 100k, можно увеличить до миллионов
    
    full_results, full_best_score, full_best_params = run_mega_search(
        max_combinations=max_combinations, 
        save_results=True, 
        create_plots=True
    )
    
    print(f"\n" + "=" * 80)
    print("📊 MEGA SEARCH SUMMARY")
    print("=" * 80)
    print(f"🎯 Best Accuracy: {full_best_score:.2f}%")
    print(f"🔧 Best Parameters:")
    print(f"   - Hidden Units: {full_best_params[0]}")
    print(f"   - Learning Rate: {full_best_params[1]}")
    print(f"   - Epochs: {full_best_params[2]}")
    print(f"   - Activation: {full_best_params[3]}")
    print(f"   - Noise: {full_best_params[4]}")
    print(f"   - Test Ratio: {full_best_params[5]}")
    print(f"   - Batch Size: {full_best_params[6]}")
    print(f"   - Momentum: {full_best_params[7]}")
    
    # Анализ результатов
    accuracies = [r['accuracy'] for r in full_results]
    print(f"\n📈 Statistics:")
    print(f"   - Mean Accuracy: {np.mean(accuracies):.2f}%")
    print(f"   - Std Accuracy: {np.std(accuracies):.2f}%")
    print(f"   - Min Accuracy: {np.min(accuracies):.2f}%")
    print(f"   - Max Accuracy: {np.max(accuracies):.2f}%")
    print(f"   - Combinations tested: {len(full_results):,}")
    print(f"   - Coverage: {len(full_results)/total_combinations*100:.6f}%")
    
    print(f"\n✅ MEGA search completed! Tested {len(full_results):,} combinations out of {total_combinations:,} possible!")
    print(f"💡 To test more combinations, increase max_combinations in the code.") 