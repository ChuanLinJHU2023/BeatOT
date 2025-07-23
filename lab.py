from models import *
from visualizations import *
from evaluate import *
import sys
import os
import time
from scipy.spatial import distance


method_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
project_root = "./"

# Step 1: Get data
case_number = 0
data = np.load(project_root + f'cases/case{case_number}.npz')
X_source = data['X_source']
y_source = data['y_source']
X_target = data['X_target']
y_target = data['y_target']


# Step 2: Set Hyper Parameter
hyper_parameter_p = 2
hyper_parameter_c = 2
order_parameter_p = 2
learning_rate = 0.001
num_epochs = 100000
num_prints = 10
num_epochs_per_print = num_epochs // num_prints
num_hidden_units = 16
list_of_num_hidden_units = [num_hidden_units]
model = SimpleClassifier(list_of_num_hidden_units)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


cost_matrix_X = np.zeros((X_source.shape[0], X_target.shape[0]))
for i in range(X_source.shape[0]):
    for j in range(X_target.shape[0]):
        cost_matrix_X[i, j] = np.linalg.norm(X_source[i] - X_target[j], ord=hyper_parameter_p) ** hyper_parameter_p


cost_matrix_X2 = distance.cdist(X_source, X_target, metric='minkowski', p=order_parameter_p)**order_parameter_p
assert(np.allclose(cost_matrix_X, cost_matrix_X2))
