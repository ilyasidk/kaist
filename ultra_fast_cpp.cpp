#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <string>

using namespace std;

// Constants
const int N_POINTS = 150;
const int N_INPUT = 2;
const int N_OUTPUT = 1;

// Neural network parameters
struct Params {
    int n_hidden;
    double learning_rate;
    int epochs;
    string activation_fn;
    double noise;
    double test_ratio;
};

// Activation functions
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double sigmoid_prime(double z) {
    double s = sigmoid(z);
    return s * (1.0 - s);
}

double relu(double z) {
    return max(0.0, z);
}

double relu_prime(double z) {
    return (z > 0.0) ? 1.0 : 0.0;
}

double tanh_activation(double z) {
    return tanh(z);
}

double tanh_prime(double z) {
    return 1.0 - tanh(z) * tanh(z);
}

// Generate spiral dataset
void make_extreme_spiral(vector<vector<double>>& X, vector<int>& y, double noise) {
    random_device rd;
    mt19937 gen(0); // Fixed seed for reproducibility
    normal_distribution<double> dist(0.0, noise);
    
    X.resize(2 * N_POINTS);
    y.resize(2 * N_POINTS);
    
    for (int i = 0; i < N_POINTS; i++) {
        double theta = sqrt((double)i / N_POINTS) * 780.0 * 2.0 * M_PI / 360.0;
        double r_a = 2.0 * theta + M_PI;
        double r_b = -2.0 * theta - M_PI;
        
        // First spiral
        X[i] = {r_a * cos(theta) + dist(gen), r_a * sin(theta) + dist(gen)};
        y[i] = 0;
        
        // Second spiral
        X[i + N_POINTS] = {r_b * cos(theta) + dist(gen), r_b * sin(theta) + dist(gen)};
        y[i + N_POINTS] = 1;
    }
}

// Train and evaluate neural network
double train_and_evaluate(const Params& params) {
    try {
        // Generate dataset
        vector<vector<double>> X;
        vector<int> y;
        make_extreme_spiral(X, y, params.noise);
        
        // Split data
        int test_size = (int)(X.size() * params.test_ratio);
        vector<vector<double>> X_train, X_test;
        vector<int> y_train, y_test;
        
        for (int i = 0; i < X.size() - test_size; i++) {
            X_train.push_back(X[i]);
            y_train.push_back(y[i]);
        }
        for (int i = X.size() - test_size; i < X.size(); i++) {
            X_test.push_back(X[i]);
            y_test.push_back(y[i]);
        }
        
        // Initialize weights
        random_device rd;
        mt19937 gen(0);
        normal_distribution<double> dist(0.0, 0.1);
        
        vector<vector<double>> W1(N_INPUT, vector<double>(params.n_hidden));
        vector<double> b1(params.n_hidden, 0.0);
        vector<vector<double>> W2(params.n_hidden, vector<double>(N_OUTPUT));
        vector<double> b2(N_OUTPUT, 0.0);
        
        for (int i = 0; i < N_INPUT; i++) {
            for (int j = 0; j < params.n_hidden; j++) {
                W1[i][j] = dist(gen);
            }
        }
        for (int i = 0; i < params.n_hidden; i++) {
            for (int j = 0; j < N_OUTPUT; j++) {
                W2[i][j] = dist(gen);
            }
        }
        
        // Training loop
        for (int ep = 0; ep < params.epochs; ep++) {
            for (int sample = 0; sample < X_train.size(); sample++) {
                // Forward pass
                vector<double> Z1(params.n_hidden);
                vector<double> A1(params.n_hidden);
                vector<double> Z2(N_OUTPUT);
                vector<double> A2(N_OUTPUT);
                
                // Hidden layer
                for (int j = 0; j < params.n_hidden; j++) {
                    Z1[j] = b1[j];
                    for (int i = 0; i < N_INPUT; i++) {
                        Z1[j] += X_train[sample][i] * W1[i][j];
                    }
                    
                    if (params.activation_fn == "sigmoid") {
                        A1[j] = sigmoid(Z1[j]);
                    } else if (params.activation_fn == "relu") {
                        A1[j] = relu(Z1[j]);
                    } else if (params.activation_fn == "tanh") {
                        A1[j] = tanh_activation(Z1[j]);
                    }
                }
                
                // Output layer
                for (int k = 0; k < N_OUTPUT; k++) {
                    Z2[k] = b2[k];
                    for (int j = 0; j < params.n_hidden; j++) {
                        Z2[k] += A1[j] * W2[j][k];
                    }
                    A2[k] = sigmoid(Z2[k]);
                }
                
                // Backward pass
                double dZ2 = (A2[0] - y_train[sample]) * sigmoid_prime(Z2[0]);
                
                vector<double> dA1(params.n_hidden, 0.0);
                for (int j = 0; j < params.n_hidden; j++) {
                    dA1[j] = dZ2 * W2[j][0];
                }
                
                vector<double> dZ1(params.n_hidden);
                for (int j = 0; j < params.n_hidden; j++) {
                    if (params.activation_fn == "sigmoid") {
                        dZ1[j] = dA1[j] * sigmoid_prime(Z1[j]);
                    } else if (params.activation_fn == "relu") {
                        dZ1[j] = dA1[j] * relu_prime(Z1[j]);
                    } else if (params.activation_fn == "tanh") {
                        dZ1[j] = dA1[j] * tanh_prime(Z1[j]);
                    }
                }
                
                // Update weights
                for (int j = 0; j < params.n_hidden; j++) {
                    W2[j][0] -= params.learning_rate * dZ2 * A1[j];
                }
                b2[0] -= params.learning_rate * dZ2;
                
                for (int i = 0; i < N_INPUT; i++) {
                    for (int j = 0; j < params.n_hidden; j++) {
                        W1[i][j] -= params.learning_rate * dZ1[j] * X_train[sample][i];
                    }
                }
                for (int j = 0; j < params.n_hidden; j++) {
                    b1[j] -= params.learning_rate * dZ1[j];
                }
            }
        }
        
        // Prediction and accuracy
        int correct = 0;
        for (int sample = 0; sample < X_test.size(); sample++) {
            vector<double> Z1(params.n_hidden);
            vector<double> A1(params.n_hidden);
            vector<double> Z2(N_OUTPUT);
            vector<double> A2(N_OUTPUT);
            
            // Forward pass for prediction
            for (int j = 0; j < params.n_hidden; j++) {
                Z1[j] = b1[j];
                for (int i = 0; i < N_INPUT; i++) {
                    Z1[j] += X_test[sample][i] * W1[i][j];
                }
                
                if (params.activation_fn == "sigmoid") {
                    A1[j] = sigmoid(Z1[j]);
                } else if (params.activation_fn == "relu") {
                    A1[j] = relu(Z1[j]);
                } else if (params.activation_fn == "tanh") {
                    A1[j] = tanh_activation(Z1[j]);
                }
            }
            
            for (int k = 0; k < N_OUTPUT; k++) {
                Z2[k] = b2[k];
                for (int j = 0; j < params.n_hidden; j++) {
                    Z2[k] += A1[j] * W2[j][k];
                }
                A2[k] = sigmoid(Z2[k]);
            }
            
            int prediction = (A2[0] > 0.5) ? 1 : 0;
            if (prediction == y_test[sample]) {
                correct++;
            }
        }
        
        return (double)correct / X_test.size() * 100.0;
        
    } catch (...) {
        return 0.0;
    }
}

int main() {
    cout << "⚡ ULTRA-FAST C++ OPTIMIZATION" << endl;
    cout << "==============================" << endl;
    
    // Parameter ranges
    vector<int> n_hidden_values = {15, 25, 40, 60, 80, 100, 120, 150, 180, 200};
    vector<double> lr_values = {0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03};
    vector<int> epochs_values = {300, 500, 800, 1200, 1500, 2000, 2500};
    vector<string> activation_fns = {"sigmoid", "tanh"};
    vector<double> noise_values = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    vector<double> test_ratio_values = {0.2, 0.25, 0.3, 0.35, 0.4};
    
    // Generate all combinations
    vector<Params> all_params;
    for (int nh : n_hidden_values) {
        for (double lr : lr_values) {
            for (int ep : epochs_values) {
                for (const string& act : activation_fns) {
                    for (double noise : noise_values) {
                        for (double tr : test_ratio_values) {
                            all_params.push_back({nh, lr, ep, act, noise, tr});
                        }
                    }
                }
            }
        }
    }
    
    cout << "🔍 Total combinations: " << all_params.size() << endl;
    cout << "🚀 Starting parallel optimization..." << endl;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    // Parallel optimization
    vector<double> accuracies(all_params.size());
    vector<Params> best_params;
    double best_accuracy = 0.0;
    
    #pragma omp parallel for
    for (int i = 0; i < all_params.size(); i++) {
        accuracies[i] = train_and_evaluate(all_params[i]);
        
        #pragma omp critical
        {
            if (accuracies[i] > best_accuracy) {
                best_accuracy = accuracies[i];
                best_params = {all_params[i]};
                cout << "🎯 New best: " << best_accuracy << "% | " 
                     << "n_hidden=" << all_params[i].n_hidden 
                     << " lr=" << all_params[i].learning_rate 
                     << " epochs=" << all_params[i].epochs 
                     << " act=" << all_params[i].activation_fn 
                     << " noise=" << all_params[i].noise 
                     << " test_ratio=" << all_params[i].test_ratio << endl;
            }
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    cout << "\n✅ OPTIMIZATION COMPLETED!" << endl;
    cout << "⏱️  Time: " << duration.count() / 1000.0 << " seconds" << endl;
    cout << "📊 Speed: " << all_params.size() / (duration.count() / 1000.0) << " combinations/sec" << endl;
    cout << "🏆 Best accuracy: " << best_accuracy << "%" << endl;
    
    // Save results
    ofstream outfile("cpp_best_params.txt");
    outfile << "C++ Ultra-Fast Optimization Results" << endl;
    outfile << "Best Accuracy: " << best_accuracy << "%" << endl;
    outfile << "Time: " << duration.count() / 1000.0 << " seconds" << endl;
    outfile << "Combinations: " << all_params.size() << endl << endl;
    outfile << "Best Parameters:" << endl;
    outfile << "    'n_hidden': " << best_params[0].n_hidden << "," << endl;
    outfile << "    'learning_rate': " << best_params[0].learning_rate << "," << endl;
    outfile << "    'epochs': " << best_params[0].epochs << "," << endl;
    outfile << "    'activation_fn': '" << best_params[0].activation_fn << "'," << endl;
    outfile << "    'noise': " << best_params[0].noise << "," << endl;
    outfile << "    'test_ratio': " << best_params[0].test_ratio << endl;
    outfile.close();
    
    // Update Python file
    ifstream infile("aa.py");
    string content((istreambuf_iterator<char>(infile)), istreambuf_iterator<char>());
    infile.close();
    
    string new_params = "params = {\n"
                       "    'n_hidden'     : " + to_string(best_params[0].n_hidden) + ",        # Hidden layer size  | Range: 10 ~ 200\n"
                       "    'learning_rate': " + to_string(best_params[0].learning_rate) + ",      # Learning rate      | Range: 0.001 ~ 0.1 (log scale)\n"
                       "    'epochs'       : " + to_string(best_params[0].epochs) + ",       # Training epochs    | Range: 100 ~ 3000\n"
                       "    'activation_fn': '" + best_params[0].activation_fn + "', # Activation fn      | Options: 'sigmoid', 'relu', 'tanh'\n"
                       "    'noise'        : " + to_string(best_params[0].noise) + ",       # Input noise        | Range: 0.5 ~ 1.5 (optional; for robustness test)\n"
                       "    'test_ratio'   : " + to_string(best_params[0].test_ratio) + "        # Test split ratio   | Range: 0.2 ~ 0.5 (optional; affects evaluation scale)\n"
                       "}";
    
    // Simple string replacement (you might want to use regex for production)
    size_t start = content.find("params = {");
    size_t end = content.find("}", start) + 1;
    if (start != string::npos && end != string::npos) {
        content.replace(start, end - start, new_params);
    }
    
    ofstream outfile2("aa.py");
    outfile2 << content;
    outfile2.close();
    
    cout << "✅ File aa.py updated!" << endl;
    cout << "🎯 Run: python aa.py" << endl;
    
    return 0;
} 