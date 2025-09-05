import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn

# ---------------------------
# Step 1: Create a multi-class dataset
# ---------------------------
NUM_CLASSES = 4          # number of classes
NUM_FEATURES = 2         # number of features per data point
RANDOM_SEED = 42         # for reproducibility

# Generate blobs (clusters of data)
X_blob, y_blob = make_blobs(
    n_samples=1000,          # total samples
    n_features=NUM_FEATURES, # number of input features (x, y)
    centers=NUM_CLASSES,     # number of clusters = number of classes
    cluster_std=1.5,         # spread of the clusters
    random_state=RANDOM_SEED
)

# ---------------------------
# Step 2: Convert to PyTorch tensors
# ---------------------------
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# ---------------------------
# Step 3: Split into train and test sets
# ---------------------------
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED
)

# Decide whether to use GPU (if available) or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Step 4: Build the model
# ---------------------------
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )
    def forward(self, x):
        return self.linear_layer_stack(x)

# ---------------------------
# Step 5: Accuracy function
# ---------------------------
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # number of correct predictions
    acc = (correct / len(y_pred)) * 100              # percentage accuracy
    return acc

# ---------------------------
# Step 6: Decision boundary plotting
# ---------------------------
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Prepare grid data for prediction
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Classification: multi-class → use softmax
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))

    # Reshape and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# ---------------------------
# Step 7: Create model instance, loss function, optimizer
# ---------------------------
model_4 = BlobModel(input_features=NUM_FEATURES, 
                    output_features=NUM_CLASSES, 
                    hidden_units=8).to(device)

loss_fn = nn.CrossEntropyLoss()  # suitable for multi-class classification
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.1)

# ---------------------------
# Step 8: Check outputs (logits and probabilities)
# ---------------------------
print(model_4(X_blob_train.to(device))[:5])
print(model_4(X_blob_train.to(device))[0].shape, NUM_CLASSES)

# Prediction logits
y_logits = model_4(X_blob_test.to(device))

# Convert logits → probabilities using softmax
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
print(y_pred_probs[:5])

# Probabilities always sum to 1
print(torch.sum(y_pred_probs[0]))

# Show predicted probabilities and class for the first sample
print(y_pred_probs[0])
print(torch.argmax(y_pred_probs[0]))

# ---------------------------
# Step 9: Training loop
# ---------------------------
torch.manual_seed(42)
epochs = 100

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model_4.train()

    # Forward pass
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # Compute loss and accuracy
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)

    # Zero gradients, backpropagation, optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing phase
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# ---------------------------
# Step 10: Evaluate final predictions
# ---------------------------
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = y_pred_probs.argmax(dim=1)

print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")

# ---------------------------
# Step 11: Visualize decision boundary
# ---------------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()
