import torch

from models import *
from visualizations import *
from evaluate import *
from scipy.spatial import distance
from distances import *
import sys
import os
import time
import ot


method_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
project_root = "../"


# Key Function of CD
def get_transport_matrix(X_source, y_source, X_target, y_target_prediction_tensor, hyper_parameter_p=2, hyper_parameter_c=2):
    y_target_prediction = y_target_prediction_tensor.detach().cpu().numpy()
    transport_matrix = calculate_causal_distance_between_dataset_and_soft_labelled_dataset(X_source, y_source, X_target, y_target_prediction, hyper_parameter_p, hyper_parameter_c)
    transport_matrix_tensor = torch.tensor(transport_matrix)
    return transport_matrix_tensor


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
learning_rate = 0.001
num_epochs = 100000
num_prints = 10
num_epochs_per_print = num_epochs // num_prints
num_hidden_units = 16
list_of_num_hidden_units = [num_hidden_units]
model = SimpleClassifier(list_of_num_hidden_units)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Step 3: Train and Save Classifier
start_time = time.perf_counter()
loss_values = list()
loss_values_X = list()
loss_values_Y = list()

X_source_tensor = torch.tensor(X_source, dtype=torch.float32)
y_source_tensor = torch.tensor(y_source.reshape(-1, 1), dtype=torch.float32)
X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
y_target_tensor = torch.tensor(y_target.reshape(-1, 1), dtype=torch.float32)
cost_matrix_X = distance.cdist(X_source, X_target, metric='minkowski', p = hyper_parameter_p) ** hyper_parameter_p
cost_matrix_X_tensor = torch.tensor(cost_matrix_X, dtype=torch.float32)
cost_matrix_Y1_coefficient = (y_source).reshape(-1, 1).repeat(X_target.shape[0], axis=1) * hyper_parameter_c
cost_matrix_Y2_coefficient = (1 - y_source).reshape(-1, 1).repeat(X_target.shape[0], axis=1) * hyper_parameter_c
cost_matrix_Y1_coefficient_tensor = torch.tensor(cost_matrix_Y1_coefficient, dtype=torch.float32)
cost_matrix_Y2_coefficient_tensor = torch.tensor(cost_matrix_Y2_coefficient, dtype=torch.float32)


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_target_prediction_tensor = model(X_target_tensor)
    cost_matrix_Y1_variable_tensor = (1 - y_target_prediction_tensor ).reshape(1,-1).repeat(X_source.shape[0], 1)
    cost_matrix_Y2_variable_tensor = (y_target_prediction_tensor).reshape(1,-1).repeat(X_source.shape[0], 1)
    cost_matrix_Y1_tensor = cost_matrix_Y1_coefficient_tensor * cost_matrix_Y1_variable_tensor
    cost_matrix_Y2_tensor = cost_matrix_Y2_coefficient_tensor * cost_matrix_Y2_variable_tensor
    cost_matrix_tensor = cost_matrix_X_tensor + cost_matrix_Y1_tensor + cost_matrix_Y2_tensor
    transport_matrix_tensor = get_transport_matrix(X_source, y_source, X_target, y_target_prediction_tensor, hyper_parameter_p, hyper_parameter_c)
    loss = torch.sum(transport_matrix_tensor * cost_matrix_tensor)
    loss.backward()
    optimizer.step()
    if epoch % num_epochs_per_print == 0:
        loss_from_X = torch.sum(transport_matrix_tensor * cost_matrix_X_tensor)
        loss_from_Y1 = torch.sum(transport_matrix_tensor * cost_matrix_Y1_tensor)
        loss_from_Y2 = torch.sum(transport_matrix_tensor * cost_matrix_Y2_tensor)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f} (X: {loss_from_X.item():.4f}, Y1: {loss_from_Y1.item():.4f}, Y2: {loss_from_Y2.item():.4f})")
        loss_values.append(loss.item())
        loss_values_X.append(loss_from_X.item())
        loss_values_Y.append(loss_from_Y1.item() + loss_from_Y2.item())


save_directory = project_root + "checkpoints/"
save_file = get_timestamp_filename() + ".pth"
torch.save(model.state_dict(), save_directory + save_file)
end_time = time.perf_counter()
duration = end_time - start_time


# Step 4: Evaluate and record
accuracy, positive_accuracy, negative_accuracy, positive_predictions_ratio = \
    evaluate_and_print_for_binary_classification(X_target, y_target, model)

summarize_text = get_variable_names_and_values(save_file, method_name, case_number, duration, learning_rate, num_epochs, num_hidden_units,
                                               accuracy, positive_accuracy, negative_accuracy, positive_predictions_ratio, loss_values, loss_values_X, loss_values_Y)
print(summarize_text)
log_directory = project_root + "logs/"
log_file = get_timestamp_filename(just_day=True)+".txt"
with open(log_directory + log_file, 'a') as file:
    file.write(summarize_text+"\n")

visualize_domains([X_source, X_target], [y_source, y_target],
                  ['Source Domain', "Target Domain"],
                  x_limit=(-2.5, 3.5), y_limit=(-3, 3), with_model=model)