import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error
from sklearn.model_selection import KFold, train_test_split
from featurewiz import FeatureWiz
import torch
import torch.nn as nn
import torch.optim as optim

# Load and preprocess the data
data = pd.read_csv(r'C:\Users\hankishan\Desktop\dataset\29may23\syst_care_demo.csv')
data = data.fillna(data.mean())
y = data[['Syst']]
X = data.drop(columns=['Syst', 'Diast', 'timeCare', 'timePro1', 'PAT_ID', 'Syst-2', 'Diast-2'], axis=1)

# Feature transformation and selection (using FeatureWiz)
features = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=2)
X = features.fit_transform(X, y)

# Normalize the input data
X = (X - X.mean()) / X.std()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
class DiffusionModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(DiffusionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Extract the last time step output
        lstm_last_output = lstm_out[:, -1, :]
        # Reshape to match the input shape expected by the fully connected layer
        lstm_last_output = lstm_last_output.view(lstm_last_output.size(0), -1)
        # Pass through the fully connected layer
        x = self.fc(lstm_last_output)
        return x
def kfold_cross_validation(X, y, model, criterion, optimizer, num_epochs, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    r2_scores = []
    pearson_corr_values = []
    mae_scores = []
    explained_variance_scores = []
    max_error_values = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        # Ensure that X_test_tensor has three dimensions
        X_test_tensor = X_test_tensor.unsqueeze(1)
        # Ensure that X_train_tensor has three dimensions
        X_train_tensor = X_train_tensor.unsqueeze(1)

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)

        mse = mean_squared_error(y_test.values, predictions.numpy())
        r2 = r2_score(y_test.values, predictions.numpy())
        pearson_corr, _ = pearsonr(y_test.values.flatten(), predictions.numpy().flatten())
        mae = mean_absolute_error(y_test.values, predictions.numpy())
        explained_variance = explained_variance_score(y_test.values, predictions.numpy())
        max_err = max_error(y_test.values, predictions.numpy())
        mse_scores.append(mse)
        r2_scores.append(r2)
        pearson_corr_values.append(pearson_corr)
        mae_scores.append(mae)
        explained_variance_scores.append(explained_variance)
        max_error_values.append(max_err)
        print("Mean Squared Error:", mse)
        print("R-squared Score:", r2)
        print("Pearson Correlation Coefficient:", pearson_corr)
        print("Mean Absolute Error:", mae)
        print("Explained Variance Score:", explained_variance)
        print("Max Error:", max_err)

    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_pearson_corr = np.mean(pearson_corr_values)
    avg_mae = np.mean(mae_scores)
    avg_explained_variance = np.mean(explained_variance_scores)
    avg_max_error = np.mean(max_error_values)
    return avg_mse, avg_r2, avg_pearson_corr, avg_mae, avg_explained_variance, avg_max_error

# Initialize the simplified Diffusion Model
print("X_train shape:", X_train.shape)
input_size = X_train.shape[1]  # Assuming you want to use the second dimension
diffusion_model = DiffusionModel(input_size=input_size)

# Set the number of training epochs
num_epochs = 200
# Set up the optimizer and loss function
optimizer = optim.SGD(diffusion_model.parameters(), lr=0.01)
criterion = nn.MSELoss()
# Perform k-fold cross-validation
# Perform k-fold cross-validation
avg_mse, avg_r2, avg_pearson_corr, avg_mae, avg_explained_variance, avg_max_error = kfold_cross_validation(X, y, diffusion_model, criterion, optimizer, num_epochs)

# Print the average results
print("Average Mean Squared Error:", avg_mse)
print("Average R-squared Score:", avg_r2)
print("Average Pearson Correlation Coefficient:", avg_pearson_corr)
print("Average Mean Absolute Error:", avg_mae)
print("Average Explained Variance Score:", avg_explained_variance)
print("Average Max Error:", avg_max_error)