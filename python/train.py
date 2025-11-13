import os
import warnings

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleClassifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Create a model object
model = SimpleClassifier()

# Create objects for backpropagation and gradient descent
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# Train the model
epochs = 20
for epoch in range(epochs):
    for images, labels in train_loader:  # 60000 / 64 iterations in 1 epoch
        # Write code for training
        ## Hint: use ".view(images.size(0), -1)" to flatten the input images
        optimizer.zero_grad()
        pred = model(images.view(images.size(0), -1))
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

    if loss.item() < 0.5:
        break

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the model with the test dataset
model.eval()
correct = 0
total = len(test_loader.dataset)

for images, labels in test_loader:
    pred = model(images.view(images.size(0), -1))

    # Postprocessing: hardmax is used for inference
    _, y = torch.max(pred, 1)

    # Accumulate the number of correct predictions
    correct += (y == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}% " f"({correct}/{total} correct)")

# Prepare a dummy input for ONNX export
sample_image, _ = train_data[0]
num_features = (
    sample_image.numel()
)  ## Hint: use ".numel()" to get the total number of features
dummy_input = torch.randn(1, num_features)

# Export to ONNX
# Call torch.onnx.export
torch.onnx.export(
    model,
    dummy_input,
    "./models/simple_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
)

print("Successfully saved simple_classifier.onnx in ./models")
