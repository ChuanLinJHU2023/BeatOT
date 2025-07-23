from models import *
from visualizations import *
from evaluate import *

# Step 1: Get data
case_number = 0
data = np.load(f'../cases/case{case_number}.npz')
X_source = data['X_source']
y_source = data['y_source']
X_target = data['X_target']
y_target = data['y_target']


# Step 2: Set Hyper Parameter
learning_rate = 0.001
num_epochs = 30000
num_epochs_per_print = num_epochs/10
list_of_num_hidden_units = [16]
model = SimpleClassifier(list_of_num_hidden_units)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Step 3: Train Classifier
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



# Step 4: Visualize
print_hyper_parameters(len(X_source), None, None, learning_rate, num_epochs, list_of_num_hidden_units)
accuracy, positive_accuracy, negative_accuracy, positive_predictions_ratio = \
    evaluate_and_print_for_binary_classification(X_target, y_target, model)
visualize_domains([X_source, X_target], [y_source, y_target],
                  ['Source Domain', "Target Domain"],
                  x_limit=(-2.5, 3.5), y_limit=(-3, 3), with_model=model)
log_file = "log_tradition.txt"
with open(log_file, 'a') as file:
    summarization = f"Learning Rate {learning_rate}; Accuracy: {accuracy}\n"
    file.write(summarization)
    print(summarization)