import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

mapping = {'1': 1, '2': -1, '0': 0}
reward_Map = {'positive': 1, 'negative': -1}

class TTTDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        self.data = []
        for line in lines:
            board = line.strip().split(',')
            state = [mapping[c] for c in board[:-1]]
            reward = reward_Map[board[-1]]
            self.data.append((state, reward))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, reward = self.data[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(reward, dtype=torch.float32)


class DQL(nn.Module):
    def __init__(self, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super(DQL, self).__init__()
        self.fc1 = nn.Linear(9, 32)
        self.out = nn.Linear(32, 9)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)  
    
    def select_action(self, state):
        if torch.rand(1).item() < self.epsilon:
            # Random action (exploration)
            return torch.randint(0, 9, (1,)).item()
        else:
            # Best action based on Q-values (exploitation)
            with torch.no_grad():
                q_values = self(state)
                return torch.argmax(q_values).item()

    def best_action(self, state, board):
        with torch.no_grad():
            q_values = self(state)

        q_values = q_values.squeeze(0).cpu().numpy()
            
        # Select the action with the highest Q-value
        valid_actions = np.where(board.flatten() == 0)[0]
        valid_q_values = q_values[valid_actions]
        action = valid_actions[np.argmax(valid_q_values)]
        return action
         
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def trainDataset(model, optimizer, criterion, epochs=10, batch_size=32, lr=0.001):
        dataset = TTTDataset('trainTTT.data')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for state, reward in dataloader:
                q_values = model(state)
                target = q_values.clone().detach()
                for i in range(state.size(0)):
                    for j in range(9):
                        if state[i][j] == 0:
                            target[i][j] = reward[i]
                        else:
                            target[i][j] = q_values[i][j]

                loss = criterion(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        return model


    def train_model(self, optimizer, criterion, replay_memory, batch_size, epoch):
        if len(replay_memory) < batch_size:
            return
        for x in range(epoch):
            transitions = replay_memory.sample(batch_size)
            batch = list(zip(*transitions))
            states, actions, rewards, next_states, dones = batch

            states = torch.stack(states).squeeze(1)
            next_states = torch.stack(next_states).squeeze(1)
            
            # Convert to proper tensor shapes
            actions = torch.tensor(actions, dtype=torch.int64)
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)  

            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Get Q-values from the model
            q_values = self(states).gather(1, actions).squeeze(1)  
            next_q_values = self(next_states).max(1)[0]        
            expected_q_values = rewards + (1 - dones) * 0.99 * next_q_values

            # Compute loss
            loss = criterion(q_values, expected_q_values)
            print(loss)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def state_to_dqn_input(state, num_states=9):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
