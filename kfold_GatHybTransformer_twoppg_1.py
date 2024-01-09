import itertools

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, explained_variance_score, r2_score
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from matplotlib import pyplot as plt

# Veri okuma
data = pd.read_csv(r'C:\Users\hankishan\Desktop\dataset\29may23\syst_care_demo.csv')
y = data['Syst']
X = data.drop(columns=['Syst', 'Diast', 'timeCare', 'timePro1', 'PAT_ID', 'Syst-2', 'Diast-2'], axis=1)

# Eğitim veri setinde NaN değerleri kontrol etme
print(data.isnull().sum())

# NaN değerleri temizleme
data = data.dropna()

# StandardScaler kullanarak veriyi ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bağımsız değişken sayısını belirtme
num_features = X_scaled.shape[1]

# Assuming X_scaled.shape[0] is the number of nodes in your graph
num_nodes = X_scaled.shape[0]
# Create edge_index using torch_geometric.utils.add_self_loops
edge_index = add_self_loops(torch.zeros(2, num_nodes, dtype=torch.long), num_nodes=num_nodes)[0]

# Hyperparameter grid
learning_rates = [0.0001]
num_transformer_layers_values = [2]
batch_sizes = [16]

# Initialize variables to store the best hyperparameters and corresponding performance
best_hyperparameters = {}
best_mae = float('inf')  # Initialize with a large value
# Lists to store training and test losses during each epoch
train_losses = []
test_losses = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y.values).view(-1, 1)  # Fix the target dimension

# Number of folds for K-Fold Cross Validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
# Initialize lists to store predictions for each fold
all_predictions = []
# Perform K-Fold Cross Validation
for fold, (train_indices, test_indices) in enumerate(kf.split(X_tensor)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Split data into training and test sets for the current fold
    X_train, X_test = X_tensor[train_indices], X_tensor[test_indices]
    y_train, y_test = y_tensor[train_indices], y_tensor[test_indices]

    # TensorDataset and DataLoader for the current fold
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    batch_size=32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Perform grid search
    for lr, num_layers, batch_size in itertools.product(learning_rates, num_transformer_layers_values, batch_sizes):
        class GraphTransformerModel(nn.Module):
            def __init__(self, edge_index):
                super(GraphTransformerModel, self).__init__()
                self.embedding = nn.Linear(num_features, 16)
                self.gat_layer = GATConv(in_channels=16, out_channels=16, heads=8, edge_index=edge_index)
                # Determine the output size of GATConv layer
                gat_output_size = 16 * 8  # Assuming 8 heads
                self.transformer_layer = nn.TransformerEncoderLayer(d_model=gat_output_size, nhead=8, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
                self.fc = nn.Linear(gat_output_size, 1)

            def forward(self, x, edge_index):
                # print("Input Shape:", x.shape)
                x = self.embedding(x)
                # print("After Embedding Layer Shape:", x.shape)
                x = self.gat_layer(x, edge_index=edge_index).flatten(1)
                # print("After GAT Layer Shape:", x.shape)
                x = self.transformer_encoder(x)
                # print("After Transformer Encoder Shape:", x.shape)
                x = self.fc(x.flatten(1))  # Flatten the output of TransformerEncoder before passing to nn.Linear
                # print("After FC Layer Shape:", x.shape)
                return x

        # Modeli oluşturup GPU'ya taşıma (eğer kullanılabilirse)
        model = GraphTransformerModel(edge_index=edge_index).to(device)
        # Kayıp fonksiyonu ve optimizasyon fonksiyonunu tanımlama
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        # Modeli eğitme
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(X, edge_index=edge_index)
                loss = criterion(output, y.view(-1, 1))
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # Average training loss for the epoch
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)

            # Modeli test etme
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                fold_predictions = []  # Initialize list to store predictions for the current fold
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    output = model(X, edge_index=edge_index)
                    test_loss += criterion(output, y.view(-1, 1)).item()
                    fold_predictions.append(output)

                average_test_loss = test_loss / len(test_loader)
                test_losses.append(average_test_loss)
                # Store predictions for the current fold
                fold_predictions = torch.cat(fold_predictions, dim=0)
                all_predictions.append(fold_predictions)

            average_test_loss = test_loss / len(test_loader)
            test_losses.append(average_test_loss)
            # Calculate metrics
            mae = mean_absolute_error(y_test.cpu().numpy(), all_predictions[-1].cpu().numpy())
            mse = mean_squared_error(y_test.cpu().numpy(), all_predictions[-1].cpu().numpy())
            max_err = max_error(y_test.cpu().numpy(), all_predictions[-1].cpu().numpy())
            exp_var = explained_variance_score(y_test.cpu().numpy(), all_predictions[-1].cpu().numpy())
            # Calculate R-squared and Pearson correlation coefficients
            y_true = y_test.cpu().numpy()
            y_pred = all_predictions[-1].cpu().numpy()
            r2 = r2_score(y_true, y_pred)
            pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            # Print or log the training and test losses for each epoch
            print(
                f"Epoch {epoch + 1}/{num_epochs} => Training Loss: {epoch_train_loss:.4f}, Test Loss: {average_test_loss:.4f}, "
                f"MAE: {mae:.4f}, MSE: {mse:.4f}, Max Error: {max_err:.4f}, Explained Variance: {exp_var:.4f}, "
                f"R-squared: {r2:.4f}, Pearson Correlation: {pearson_corr:.4f}")

            # Check if the current set of hyperparameters results in a better performance
            if mae < best_mae:
                best_mae = mae
                best_hyperparameters = {'lr': lr, 'num_layers': num_layers, 'batch_size': batch_size}
                # Save the best model
                model_best = model
                optimizer_best = optimizer
                criterion_best = criterion

# Corrected optimizer initialization for the best model
optimizer_best = optim.AdamW(model_best.parameters(), lr=best_hyperparameters['lr'])

# Train the model with the best hyperparameters on the entire dataset
for epoch in range(num_epochs):
    model_best.train()
    epoch_train_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer_best.zero_grad()
        output = model_best(X, edge_index=edge_index)  # Corrected edge_index usage
        loss = criterion_best(output, y.view(-1, 1))
        loss.backward()
        optimizer_best.step()
        epoch_train_loss += loss.item()

# Evaluate the model with the best hyperparameters on the entire dataset
model_best.eval()
with torch.no_grad():
    test_loss_best = 0.0
    all_predictions_best = []
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        output = model_best(X, edge_index=edge_index)  # Corrected edge_index usage
        test_loss_best += criterion_best(output, y.view(-1, 1)).item()
        all_predictions_best.append(output)

# Concatenate predictions from all folds
all_predictions_combined = np.concatenate(all_predictions, axis=0)

# Calculate R-squared and Pearson correlation coefficients for the best hyperparameters
y_true_best = y_tensor.cpu().numpy()
y_pred_best = torch.cat(all_predictions_best, dim=0).cpu().numpy()
# Calculate R-squared and Pearson correlation coefficients for the combined predictions
r2_combined = r2_score(y_true_best, all_predictions_combined)
pearson_corr_combined, _ = pearsonr(y_true_best.flatten(), all_predictions_combined.flatten())
# Print or use the combined metrics
print("Combined R-squared:", r2_combined)
print("Combined Pearson Correlation:", pearson_corr_combined)

r2_best = r2_score(y_true_best, y_pred_best)
pearson_corr_best, _ = pearsonr(y_true_best.flatten(), y_pred_best.flatten())

# Plot training and test losses
plt.figure(figsize=(10, 5))
epochs_range = range(1, num_epochs + 1)

# Plot only the epochs where the training losses were recorded
plt.plot(epochs_range[:len(train_losses)], train_losses, label='Training Loss')
plt.plot(epochs_range[:len(test_losses)], test_losses, label='Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.show()

# Print the best hyperparameters and performance
print("Best Hyperparameters:", best_hyperparameters)
print("Best MAE:", best_mae)
print("Best R-squared:", r2_best)
print("Best Pearson Correlation:", pearson_corr_best)
