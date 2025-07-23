import numpy as np
from models import *
from problems import *
import ot
from scipy.spatial import distance
from visualizations import *
from utils import *
from evaluate import *

# Step 1: Get data
n_samples=100
X_source, y_source, X_target, y_target = create_domain_adaptation_problem(n_samples=n_samples,  theta=0, noise_level=0.1,
                                                                          horizontally_stretch=1.5,
                                                                          pos_x_interval_source=(-np.inf, 1.7),
                                                                          pos_y_interval_source=(-np.inf, np.inf),
                                                                          neg_x_interval_source=(-np.inf, 0.2),
                                                                          neg_y_interval_source=(-np.inf, np.inf),
                                                                          pos_x_interval_target=(1.3, np.inf),
                                                                          pos_y_interval_target=(-np.inf, np.inf),
                                                                          neg_x_interval_target=(-0.2, np.inf),
                                                                          neg_y_interval_target=(-np.inf, np.inf),
                                                                          )





# Step 2: Set hyper-parameters
hyper_parameter_p = 2
hyper_parameter_c = 2
learning_rate = 0.1
num_epochs = 5000
num_epochs_per_print = num_epochs/10
list_of_num_hidden_units = [16]
model = SimpleClassifier(list_of_num_hidden_units)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Step 3: Train classifier
X_source_tensor = torch.tensor(X_source, dtype=torch.float32)
y_source_tensor = torch.tensor(y_source.reshape(-1, 1), dtype=torch.float32)
X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
y_target_tensor = torch.tensor(y_target.reshape(-1, 1), dtype=torch.float32)
cost_matrix_X = np.zeros((X_source.shape[0], X_target.shape[0]))
for i in range(X_source.shape[0]):
    for j in range(X_target.shape[0]):
        cost_matrix_X[i, j] = np.linalg.norm(X_source[i] - X_target[j], ord=hyper_parameter_p) ** hyper_parameter_p
cost_matrix_X_tensor = torch.tensor(cost_matrix_X, dtype=torch.float32)
cost_matrix_Y1_coefficient = (y_source).reshape(-1, 1).repeat(X_target.shape[0], axis=1) * hyper_parameter_c ** hyper_parameter_p
cost_matrix_Y2_coefficient = (1 - y_source).reshape(-1, 1).repeat(X_target.shape[0], axis=1) * hyper_parameter_c ** hyper_parameter_p
cost_matrix_Y1_coefficient_tensor = torch.tensor(cost_matrix_Y1_coefficient, dtype=torch.float32)
cost_matrix_Y2_coefficient_tensor = torch.tensor(cost_matrix_Y2_coefficient, dtype=torch.float32)

def get_transport_matrix_from_cost_matrix(cost_matrix_tensor):
    cost_matrix = cost_matrix_tensor.detach().numpy()
    source_distribution = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
    target_distribution = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]
    transport_matrix = ot.emd(source_distribution, target_distribution, cost_matrix)
    transport_matrix_tensor = torch.tensor(transport_matrix, dtype=torch.float32)
    return transport_matrix_tensor

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_target_prediction_tensor = model(X_target_tensor)
    cost_matrix_Y1_variable_tensor = (1 - y_target_prediction_tensor ).reshape(1,-1).repeat(X_source.shape[0], 1)
    cost_matrix_Y2_variable_tensor = (y_target_prediction_tensor).reshape(1,-1).repeat(X_source.shape[0], 1)
    cost_matrix_Y1_tensor = cost_matrix_Y1_coefficient_tensor * cost_matrix_Y1_variable_tensor
    cost_matrix_Y2_tensor = cost_matrix_Y2_coefficient_tensor * cost_matrix_Y2_variable_tensor
    cost_matrix_tensor = cost_matrix_X_tensor + cost_matrix_Y1_tensor + cost_matrix_Y2_tensor
    transport_matrix_tensor = get_transport_matrix_from_cost_matrix(cost_matrix_tensor)
    loss = torch.sum(transport_matrix_tensor * cost_matrix_tensor)
    loss.backward()
    optimizer.step()
    if epoch % num_epochs_per_print == 0:
        loss_from_X = torch.sum(transport_matrix_tensor * cost_matrix_X_tensor)
        loss_from_Y1 = torch.sum(transport_matrix_tensor * cost_matrix_Y1_tensor)
        loss_from_Y2 = torch.sum(transport_matrix_tensor * cost_matrix_Y2_tensor)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f} (X: {loss_from_X.item():.4f}, Y1: {loss_from_Y1.item():.4f}, Y2: {loss_from_Y2.item():.4f})")


# Step 4: Visualize
print_hyper_parameters(n_samples, hyper_parameter_p, hyper_parameter_c, learning_rate, num_epochs, list_of_num_hidden_units)
evaluate_and_print_for_binary_classification(X_target, y_target, model)
visualize_domains([X_source, X_target], [y_source, y_target],
                  [f'Source Domain c={hyper_parameter_c}',
                   f"Target Domain c={hyper_parameter_c}"],
                  x_limit=(-2.5, 3.5), y_limit=(-3, 3), with_model=model)