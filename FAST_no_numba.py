import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product

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

best_score, best_params = -1, None
results = []  # сохраняем все результаты

print("🚀 Starting Grid Search...")
total_combinations = len(list(product(*grid.values())))
print(f"📊 Total combinations to test: {total_combinations}")

for i, combo in enumerate(product(*grid.values())):
    acc = train_and_eval(combo)
    results.append((combo, acc))
    if acc > best_score:
        best_score, best_params = acc, combo
    print(f"Progress: {i+1}/{total_combinations} - Params={combo} -> Acc={acc:.2f}%")

print("\n✅ Best Params:", best_params)
print("🎯 Best Accuracy:", best_score)

# выводим топ-3 лучших
results.sort(key=lambda x: x[1], reverse=True)
print("\n🔥 TOP-3:")
for i in range(min(3, len(results))):
    print(f"{i+1}. Params={results[i][0]} -> Acc={results[i][1]:.2f}%") 