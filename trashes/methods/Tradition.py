from models import *
from visualizations import *
from evaluate import *
import sys
import os
import time


method_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
project_root = "../"

# Step 1: Get data
case_number = 0
data = np.load(project_root + f'cases/case{case_number}.npz')
X_source = data['X_source']
y_source = data['y_source']
X_target = data['X_target']
y_target = data['y_target']


# Step 2: Set Hyper Parameter
learning_rate = 0.001
num_epochs = 30000
num_prints = 10
num_epochs_per_print = num_epochs // num_prints
num_hidden_units = 16
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

visualize_domains([X_source, X_target], [y_source, y_target],
                  ['Source Domain', "Target Domain"],
                  x_limit=(-2.5, 3.5), y_limit=(-3, 3), with_model=model)