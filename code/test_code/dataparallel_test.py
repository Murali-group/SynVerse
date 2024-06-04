import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Print the size of the input each GPU receives
        print(f"Input batch size per GPU: {x.size()}", flush=True)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleCNN()

# Check if multiple GPUs are available and wrap the model using nn.DataParallel
# if torch.cuda.is_available() and torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs for training.")
#     model = nn.DataParallel(model)
#     print("Model is wrapped with nn.DataParallel")
# else:
#     print("Using single GPU or CPU.")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to {device}")

# Dummy dataset for demonstration
input_tensor = torch.randn(64, 3, 32, 32).to(device)  # Example input: batch size of 64 and 32x32 RGB images
target = torch.randint(0, 10, (64,)).to(device)  # Random targets for demonstration

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop for one epoch
model.train()  # Set the model to training mode
optimizer.zero_grad()  # Zero the gradients
output = model(input_tensor)  # Forward pass
loss = criterion(output, target)  # Calculate loss
loss.backward()  # Backpropagate the gradients
optimizer.step()  # Update weights

print(f"Training loss: {loss.item()}")
