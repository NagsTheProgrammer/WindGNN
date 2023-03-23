from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Test the combined GCN-GRU model
combined_model.eval()
test_outputs = combined_model(test_data, edge_index).detach().cpu().numpy()

# Calculate evaluation metrics
mae = mean_absolute_error(test_labels, test_outputs)
rmse = np.sqrt(mean_squared_error(test_labels, test_outputs))

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

# Visualize the predictions against the actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(test_labels, label='Actual')
plt.plot(test_outputs, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()