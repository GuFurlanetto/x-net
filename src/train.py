import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time

from data_handlers import COIN
from x_net.controller import Sync_AV

# Paths
data_path = Path("data/coin/test_dataset")  # Path to your dataset folder
raw_data_path = Path("data/coin/raw_videos")  # Path to raw audio data
config_path = "src/config_files/sync_av.json"  # Path to your model config file

# Hyperparameters
batch_size = 32
num_epochs = 20
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoader
train_dataset = COIN(data_path, split="train", raw_data_path=raw_data_path)
val_dataset = COIN(data_path, split="val", raw_data_path=raw_data_path)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# Model, Loss, and Optimizer
model = Sync_AV(config_path).to(device)
criterion = nn.MSELoss()  # Example loss function; adjust as needed
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# Alternatives: ReduceLROnPlateau, ExponentialLR, etc.


# Training Loop
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, (frames_with_audio, delay_seconds) in enumerate(loader):
        frames_with_audio = frames_with_audio.to(device)
        delay_seconds = delay_seconds.to(device)

        optimizer.zero_grad()
        outputs = model(frames_with_audio)  # Forward pass
        loss = criterion(outputs, delay_seconds)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

        if batch_idx % 10 == 0:  # Print info every 10 batches
            print(
                f"[Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(loader)}] "
                f"Batch Loss: {loss.item():.4f}, Running Loss: {running_loss / (batch_idx + 1):.4f}"
            )

    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch + 1} Training Completed in {elapsed_time:.2f}s")
    return running_loss / len(loader)


# Validation Loop
def validate_one_epoch(model, loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (frames_with_audio, delay_seconds) in enumerate(loader):
            frames_with_audio = frames_with_audio.to(device)
            delay_seconds = delay_seconds.to(device)

            outputs = model(frames_with_audio)
            loss = criterion(outputs, delay_seconds)

            val_loss += loss.item()

    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch + 1} Validation Completed in {elapsed_time:.2f}s")
    return val_loss / len(loader)


# Training Process
for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch + 1}/{num_epochs}")

    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch
    )
    val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch)

    print(
        f"[Epoch {epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

    # Step the scheduler
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Learning Rate for Epoch {epoch + 1}: {current_lr:.6f}")

    # Save model checkpoint
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
    print(f"Model checkpoint saved for Epoch {epoch + 1}")

print("Training Complete!")
