import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQL(nn.Module):
    def __init__(self, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super(DQL, self).__init__()
        self.fc1 = nn.Linear(27, 32)  # 1 hidden layer with 32 neurons
        self.out = nn.Linear(32, 9)  # Output layer for binary classification
        self.epsilon = epsilon  # Epsilon for exploration
        self.epsilon_decay = epsilon_decay  # Decay factor for epsilon
        self.epsilon_min = epsilon_min  # Minimum epsilon value

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        return self.out(x)  # Output layer
    
    def select_action(self, state):
        """Select an action using epsilon-greedy strategy."""
        if torch.rand(1).item() < self.epsilon:
            # Random action (exploration)
            return torch.randint(0, 9, (1,)).item()
        else:
            # Best action based on Q-values (exploitation)
            with torch.no_grad():
                q_values = self(state)
                return torch.argmax(q_values).item()
            
    def decay_epsilon(self):
        """Decay epsilon to encourage more exploitation as training progresses."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self, optimizer, criterion, replay_memory, batch_size, epoch):
        if len(replay_memory) < batch_size:
            return
        for x in range(epoch):
            transitions = replay_memory.sample(batch_size)
            batch = list(zip(*transitions))
            states, actions, rewards, next_states, dones = batch

            states = torch.stack(states).squeeze(1)  # shape: [batch_size, 27]
            next_states = torch.stack(next_states).squeeze(1)
            
            # Convert to proper tensor shapes
            actions = torch.tensor(actions, dtype=torch.int64)
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)  # Ensure shape is [batch_size, 1]

            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Get Q-values from the model
            q_values = self(states).gather(1, actions).squeeze(1)  # shape: [batch_size]
            next_q_values = self(next_states).max(1)[0]            # shape: [batch_size]
            expected_q_values = rewards + (1 - dones) * 0.99 * next_q_values

            # Compute loss
            loss = criterion(q_values, expected_q_values)
            print(loss)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_model(self, test_data_file):
        # Load the test data
        X_test, y_test = format_data(test_data_file)
        
        # Set the model to evaluation mode
        self.eval()

        # Convert data to correct shape and type
        X_test = X_test.unsqueeze(0) if len(X_test.shape) == 1 else X_test

        # Disable gradients for testing (since we don't need them)
        with torch.no_grad():
            # Get the model's predictions
            predictions = self(X_test)
            
            # Apply a sigmoid to convert the logits to probabilities
            predicted_labels = torch.sigmoid(predictions).round()  # Round to get 0 or 1
            
            # Calculate the accuracy
            correct = (predicted_labels == y_test).float()
            print(correct.sum())
            accuracy = correct.sum() / correct.size()
            
            print(f"Accuracy: {accuracy.item() * 100:.2f}%")


def format_data(data_file):
        # Map symbols to integers
        symbol_map = {'b': 0, 'x': 1, 'o': 2}
        label_map = {'positive': 1, 'negative': 0}

        X = []
        y = []

        # Open and read the file
        with open(data_file, 'r') as file:
            for line in file:
                # Strip leading/trailing whitespace and split by commas
                parts = line.strip().split(',')
                
                # Convert the board symbols to integers
                board = [symbol_map[s] for s in parts[:-1]]  # All except last part (label)
                
                # Convert the label to 0 or 1
                label = label_map[parts[-1]]  # Last part is the label
                
                # One-hot encode the board state
                one_hot_input = state_to_dqn_input(board).squeeze(0)  # shape: (27,)
                X.append(one_hot_input)
                y.append(label)

        X = torch.stack(X)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        return X, y

def state_to_dqn_input(state, num_states=9):
    # Convert board state to one-hot format: [0,0,1] for player 1, [0,1,0] for AI, [1,0,0] for empty
    one_hot = []
    for s in state:
        if s == 0: one_hot += [1, 0, 0] # Empty space
        elif s == 1: one_hot += [0, 1, 0] # player 1 (X)
        elif s == 2: one_hot += [0, 0, 1] # player 2 (O)
    return torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
