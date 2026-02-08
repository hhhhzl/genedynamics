import os
import torch
import torch.nn as nn
from torch.utils.data import random_split

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, current_state, next_state):
        x = torch.cat([current_state, next_state], dim=-1)
        return self.layers(x)
    

# def train_inverse_dynamics(model, states, next_states, actions, epochs=100, lr=0.001, batch_size=32, save_path=save_path):
#     # Check if the directory exists. If yes, delete its content
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     else:
#         for file in os.listdir(os.path.dirname(save_path)):
#             os.remove(os.path.dirname(save_path) + '/' + file)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()

#     dataset = torch.utils.data.TensorDataset(states, next_states, actions)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     for epoch in range(epochs):
#         mean_loss = 0
#         for batch_states, batch_next_states, batch_actions in loader:
#             optimizer.zero_grad()
#             predicted_actions = model(batch_states, batch_next_states)
#             loss = loss_fn(predicted_actions, batch_actions)
#             loss.backward()
#             optimizer.step()
#             mean_loss += loss.item()
#         print(f'Epoch {epoch+1}/{epochs}, loss: {mean_loss/len(loader)}')
#     torch.save(model.state_dict(), save_path)
    
def train_inverse_dynamics(model, states, next_states, actions, save_path, steps=1e5, lr=0.001, batch_size=128):
    # Check if the directory exists. If yes, delete its content
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    else:
        for file in os.listdir(os.path.dirname(save_path)):
            os.remove(os.path.dirname(save_path) + '/' + file)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(states, next_states, actions)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_test_loss = float('inf')

    epochs = int(steps / len(train_loader))
    for epoch in range(epochs):
        # Training
        model.train()
        mean_loss = 0
        for batch_states, batch_next_states, batch_actions in train_loader:
            optimizer.zero_grad()
            predicted_actions = model(batch_states, batch_next_states)
            loss = loss_fn(predicted_actions, batch_actions)
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Training loss: {mean_loss/len(train_loader)}')

        # Testing
        model.eval()
        with torch.no_grad():
            mean_test_loss = 0
            for batch_states, batch_next_states, batch_actions in test_loader:
                predicted_actions = model(batch_states, batch_next_states)
                loss = loss_fn(predicted_actions, batch_actions)
                mean_test_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Test loss: {mean_test_loss/len(test_loader)}')

        # Save the model if it performs best on the test data
        if mean_test_loss < best_test_loss:
            best_test_loss = mean_test_loss
            torch.save(model.state_dict(), save_path)
            print(f'Saving new best model')