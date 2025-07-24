from models import *
from visualizations import *
from evaluate import *
import sys
import os
import time
import ot
from distances import *

def method_tradition(project_root = None, case_number = 0, learning_rate = 0.002, num_epochs = 10000, num_prints = 10, num_hidden_units = 16):
    method_name = "Tradition"
    if project_root is None:
        project_root = "./"

    # Step 1: Get data
    data = np.load(project_root + f'cases/case{case_number}.npz')
    X_source = data['X_source']
    y_source = data['y_source']
    X_target = data['X_target']
    y_target = data['y_target']


    # Step 2: Set Hyper Parameter
    num_epochs_per_print = num_epochs // num_prints
    list_of_num_hidden_units = [num_hidden_units]
    model = SimpleClassifier(list_of_num_hidden_units)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Step 3: Train and Save Classifier
    start_time = time.perf_counter()
    loss_values = list()
    X_tensor = torch.tensor(X_source, dtype=torch.float32)
    y_tensor = torch.tensor(y_source.reshape(-1, 1), dtype=torch.float32)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % num_epochs_per_print == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
            loss_values.append(loss.item())
    save_directory = project_root + "checkpoints/"
    save_file = get_timestamp_filename() + ".pth"
    torch.save(model.state_dict(), save_directory + save_file)
    end_time = time.perf_counter()
    duration = end_time - start_time


    # Step 4: Evaluate and record
    accuracy, positive_accuracy, negative_accuracy, positive_predictions_ratio = \
        evaluate_and_print_for_binary_classification(X_target, y_target, model)

    summarize_text = get_variable_names_and_values(save_file, method_name, case_number, duration, learning_rate, num_epochs, num_hidden_units,
                                                   accuracy, positive_accuracy, negative_accuracy, positive_predictions_ratio, loss_values)
    print(summarize_text)
    log_directory = project_root + "logs/"
    log_file = get_timestamp_filename(just_day=True)+".txt"
    with open(log_directory + log_file, 'a') as file:
        file.write(summarize_text+"\n")



def method_BCD_or_CD(project_root=None, if_BCD=True, case_number=0, hyper_parameter_p=2, hyper_parameter_c=2, learning_rate=0.002,
                     num_epochs=20000, num_prints=10, num_hidden_units=16):
    method_name = "BCD" if if_BCD else "CD"
    if project_root is None:
        project_root = "./"

    # Key Function of BCD
    def get_transport_matrix_BCD(cost_matrix_tensor):
        cost_matrix = cost_matrix_tensor.detach().numpy()
        source_distribution = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
        target_distribution = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]
        transport_matrix = ot.emd(source_distribution, target_distribution, cost_matrix)
        transport_matrix_tensor = torch.tensor(transport_matrix, dtype=torch.float32)
        return transport_matrix_tensor

    def get_transport_matrix_CD(X_source, y_source, X_target, y_target_prediction_tensor, hyper_parameter_p=2,
                             hyper_parameter_c=2):
        y_target_prediction = y_target_prediction_tensor.detach().cpu().numpy()
        causal_distance, transport_matrix = calculate_causal_distance_between_dataset_and_soft_labelled_dataset(
            X_source, y_source, X_target, y_target_prediction, hyper_parameter_p, hyper_parameter_c)
        transport_matrix_tensor = torch.tensor(transport_matrix)
        return transport_matrix_tensor

    # Step 1: Get data
    data = np.load(project_root + f'cases/case{case_number}.npz')
    X_source = data['X_source']
    y_source = data['y_source']
    X_target = data['X_target']
    y_target = data['y_target']

    # Step 2: Set Hyper Parameter
    num_epochs_per_print = num_epochs // num_prints
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
    cost_matrix_X = distance.cdist(X_source, X_target, metric='minkowski', p=hyper_parameter_p) ** hyper_parameter_p
    cost_matrix_X_tensor = torch.tensor(cost_matrix_X, dtype=torch.float32)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_target_prediction_tensor = model(X_target_tensor)
        cost_matrix_Y_tensor = torch.abs(y_source_tensor.reshape(-1, 1) - y_target_prediction_tensor.reshape(1,-1)) \
                               ** hyper_parameter_p * hyper_parameter_c

        cost_matrix_tensor = cost_matrix_X_tensor + cost_matrix_Y_tensor
        if if_BCD:
            transport_matrix_tensor = get_transport_matrix_BCD(cost_matrix_tensor)
        else:
            transport_matrix_tensor = get_transport_matrix_CD(X_source, y_source, X_target, y_target_prediction_tensor,
                                                           hyper_parameter_p, hyper_parameter_c)
        loss_from_X = torch.sum(transport_matrix_tensor * cost_matrix_X_tensor)
        loss_from_Y = torch.sum(transport_matrix_tensor * cost_matrix_Y_tensor)
        loss = loss_from_X + loss_from_Y
        loss.backward()
        optimizer.step()
        if epoch % num_epochs_per_print == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f} (X: {loss_from_X.item():.4f}, Y: {loss_from_Y.item():.4f})")
            loss_values.append(loss.item())
            loss_values_X.append(loss_from_X.item())
            loss_values_Y.append(loss_from_Y.item())

    save_directory = project_root + "checkpoints/"
    save_file = get_timestamp_filename() + ".pth"
    torch.save(model.state_dict(), save_directory + save_file)
    end_time = time.perf_counter()
    duration = end_time - start_time

    # Step 4: Evaluate and record
    accuracy, positive_accuracy, negative_accuracy, positive_predictions_ratio = \
        evaluate_and_print_for_binary_classification(X_target, y_target, model)

    summarize_text = get_variable_names_and_values(save_file, method_name, case_number, duration, learning_rate,
                                                   num_epochs, num_hidden_units,
                                                   accuracy, positive_accuracy, negative_accuracy,
                                                   positive_predictions_ratio, loss_values, loss_values_X,
                                                   loss_values_Y)
    print(summarize_text)
    log_directory = project_root + "logs/"
    log_file = get_timestamp_filename(just_day=True) + ".txt"
    with open(log_directory + log_file, 'a') as file:
        file.write(summarize_text + "\n")




def method_OT(project_root=None, case_number=0, learning_rate=0.001, num_epochs=30000, num_prints=10, num_hidden_units=16):
    method_name = "OT"
    if project_root==None:
        project_root = "./"

    # Key Function of OT
    def ot_domain_adaptation(X_s, X_t):
        n_source = X_s.shape[0]
        n_target = X_t.shape[0]
        M = ot.dist(X_s, X_t, metric='sqeuclidean')
        a = np.ones((n_source,)) / n_source
        b = np.ones((n_target,)) / n_target
        transport_matrix = ot.emd(a, b, M)
        row_sums = transport_matrix.sum(axis=1, keepdims=True)
        X_s_adapted = (transport_matrix / row_sums) @ X_t
        assert X_s_adapted.shape[0] == X_s.shape[0]
        return X_s_adapted, transport_matrix


    # Step 1: Get data
    data = np.load(project_root + f'cases/case{case_number}.npz')
    X_source = data['X_source']
    y_source = data['y_source']
    X_target = data['X_target']
    y_target = data['y_target']


    # Step 2: Set Hyper Parameter
    num_epochs_per_print = num_epochs // num_prints
    list_of_num_hidden_units = [num_hidden_units]
    model = SimpleClassifier(list_of_num_hidden_units)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Step 3: Train and Save Classifier
    start_time = time.perf_counter()
    loss_values = list()
    X_source_adapted = ot_domain_adaptation(X_source, X_target)[0]
    X_tensor = torch.tensor(X_source_adapted, dtype=torch.float32)
    y_tensor = torch.tensor(y_source.reshape(-1, 1), dtype=torch.float32)


    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % num_epochs_per_print == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
            loss_values.append(loss.item())
    save_directory = project_root + "checkpoints/"
    save_file = get_timestamp_filename() + ".pth"
    torch.save(model.state_dict(), save_directory + save_file)
    end_time = time.perf_counter()
    duration = end_time - start_time


    # Step 4: Evaluate and record
    accuracy, positive_accuracy, negative_accuracy, positive_predictions_ratio = \
        evaluate_and_print_for_binary_classification(X_target, y_target, model)

    summarize_text = get_variable_names_and_values(save_file, method_name, case_number, duration, learning_rate, num_epochs, num_hidden_units,
                                                   accuracy, positive_accuracy, negative_accuracy, positive_predictions_ratio, loss_values)
    print(summarize_text)
    log_directory = project_root + "logs/"
    log_file = get_timestamp_filename(just_day=True)+".txt"
    with open(log_directory + log_file, 'a') as file:
        file.write(summarize_text+"\n")


method_OT()