#Hyperparameters

# Isolation Forest
n_estimators: 100
max_samples: auto
contamination: auto
max_features: 1.0
bootstrap: false
n_jobs: -1
random_state: 42
verbose: 0

# Local Outlier Factor
n_neighbors: 20
algorithm: auto
leaf_size: 30
metric: minkowski
p: 2
metric_params: null
contamination: auto
novelty: true
n_jobs: -1

# One-Class SVM
kernel: rbf
degree: 3
gamma: scale
coef0: 0.0
tol: 0.001
nu: 0.05
shrinking: true
cache_size: 200
verbose: false
max_iter: -1
