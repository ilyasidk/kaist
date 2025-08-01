import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===== 1) Trainable Hyperparameters Only =====
params = {
    'n_hidden'     : 22,        # Hidden layer size  | Range: 10 ~ 200
    'learning_rate': 0.0030,      # Learning rate      | Range: 0.001 ~ 0.1 (log scale)
    'epochs'       : 300,       # Training epochs    | Range: 100 ~ 3000
    'activation_fn': 'sigmoid', # Activation fn      | Options: 'sigmoid', 'relu', 'tanh'
    'noise'        : 0.55,       # Input noise        | Range: 0.5 ~ 1.5 (optional; for robustness test)
    'test_ratio'   : 0.3        # Test split ratio   | Range: 0.2 ~ 0.5 (optional; affects evaluation scale)
}

# ===== 2) Constants (Not to beу changed) =====
N_POINTS = 150
N_INPUT  = 2
N_OUTPUT = 1

# ===== 3) Dataset Generation =====
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

X, y = make_extreme_spiral(N_POINTS, noise=params['noise'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['test_ratio'], random_state=0
)

# ===== 4) Activation Functions =====
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

# ===== 5) Weight Initialization =====
np.random.seed(0)
W1 = np.random.randn(N_INPUT, params['n_hidden']) * 0.1
b1 = np.zeros((1, params['n_hidden']))
W2 = np.random.randn(params['n_hidden'], N_OUTPUT) * 0.1
b2 = np.zeros((1, N_OUTPUT))

# ===== 6) Training Loop =====
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

# ===== 7) Prediction & Accuracy =====
def predict(X):
    A1 = activation(X.dot(W1) + b1)
    A2 = sigmoid(A1.dot(W2) + b2)
    return (A2 > 0.5).astype(int)

y_pred = predict(X_test)
accuracy = (y_pred.flatten() == y_test).mean() * 100
print(f"▶ Final Accuracy: {accuracy:.2f}%")

# ===== 8) Decision Boundary Visualization =====
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-2, X[:,0].max()+2, 300),
    np.linspace(X[:,1].min()-2, X[:,1].max()+2, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = predict(grid).reshape(xx.shape)

plt.figure(figsize=(5,5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', edgecolor='k', s=25)
plt.title(f"Decision Boundary (Acc: {accuracy:.1f}%)")
plt.xticks([]); plt.yticks([])
plt.show()

def check_param_ranges(p):
    if not (10 <= p['n_hidden'] <= 200):
        print("⚠️  'n_hidden' is out of the recommended range (10 ~ 200)")
    if not (0.001 <= p['learning_rate'] <= 0.1):
        print("⚠️  'learning_rate' is out of the recommended range (0.001 ~ 0.1)")
    if not (100 <= p['epochs'] <= 3000):
        print("⚠️  'epochs' is out of the recommended range (100 ~ 3000)")
    if p['activation_fn'] not in ['sigmoid', 'relu', 'tanh']:
        print("⚠️  'activation_fn' must be one of: 'sigmoid', 'relu', 'tanh'")
    if not (0.5 <= p['noise'] <= 1.5):
        print("⚠️  'noise' is out of the recommended range (0.5 ~ 1.5)")
    if not (0.2 <= p['test_ratio'] <= 0.5):
        print("⚠️  'test_ratio' is out of the recommended range (0.2 ~ 0.5)")

check_param_ranges(params)  # Run the range check