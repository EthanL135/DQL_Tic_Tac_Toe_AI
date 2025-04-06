import sys
from collections import deque
import pygame
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import random
import gymnasium as gym
from DQL import DQL, format_data, state_to_dqn_input

pygame.init()

# Colors
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Proportions & Sizes
WIDTH = 300
HEIGHT = 300
LINE_WIDTH = 5
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic-Tac-Toe AI')
screen.fill(BLACK)

board = np.zeros((BOARD_ROWS, BOARD_COLS))

# Draws tic-tac-toe board lines
def draw_lines(color=WHITE):
    for i in range(1, BOARD_ROWS):
        pygame.draw.line(screen, color, (0, SQUARE_SIZE * i), (WIDTH, SQUARE_SIZE * i), LINE_WIDTH)
        pygame.draw.line(screen, color, (SQUARE_SIZE * i, 0), (SQUARE_SIZE * i, HEIGHT), LINE_WIDTH)

# Draws the 'X' and 'O' figures
def draw_figures(color=WHITE):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 1:
                pygame.draw.circle(screen, color, (int(col * SQUARE_SIZE + SQUARE_SIZE // 2),
                int(row * SQUARE_SIZE + SQUARE_SIZE // 2)), CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row][col] == 2:
                pygame.draw.line(screen, color, (col * SQUARE_SIZE + SQUARE_SIZE // 4, row * SQUARE_SIZE + SQUARE_SIZE // 4),
                (col * SQUARE_SIZE + 3 * SQUARE_SIZE // 4, row * SQUARE_SIZE + 3 * SQUARE_SIZE // 4), CROSS_WIDTH)
                pygame.draw.line(screen, color, (col * SQUARE_SIZE + SQUARE_SIZE // 4, row * SQUARE_SIZE + 3 * SQUARE_SIZE // 4),
                (col * SQUARE_SIZE + 3 * SQUARE_SIZE // 4, row * SQUARE_SIZE + SQUARE_SIZE // 4), CROSS_WIDTH)

# Marks the current player on the square tile
def mark_square(row, col, player):
    board[row][col] = player

# Checks if square tile is available
def available_square(row, col):
    return board[row][col] == 0

# Checks if the game board is full
def is_board_full(check_board=board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if check_board[row][col] == 0:
                return False
    return True

# Checks to see if there's a winner
def check_win(player, check_board=board):
    for col in range(BOARD_COLS):
        if check_board[0][col] == player and check_board[1][col] == player and check_board[2][col] == player:
            return True

    for row in range(BOARD_ROWS):
        if check_board[row][0] == player and check_board[row][1] == player and check_board[row][2] == player:
            return True

    if check_board[0][0] == player and check_board[1][1] == player and check_board[2][2] == player:
        return True

    if check_board[0][2] == player and check_board[1][1] == player and check_board[2][0] == player:
        return True

    return False

def best_move():
    best_score = -1000
    move = (-1, -1)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 0:
                board[row][col] = 2
                # score = minmax(board, 0, False)
                board[row][col] = 0
                if score > best_score:
                    best_score = score
                    move = (row, col)

    if move != (-1, -1):
        mark_square(move[0], move[1], 2)
        return True
    return False

# Resets the board
def restart_game():
    screen.fill(BLACK)
    draw_lines()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board[row][col] = 0


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes) # first fully connected layer (hidden layer)
        self.out = nn.Linear(h1_nodes, out_actions) # output layer

    # Feed forward
    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply ReLU activation
        x = self.out(x) # calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, experience):
        """Store experience in memory."""
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)  # Remove oldest experience
        self.memory.append(experience)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def play_game(model, replay_memory, num_games=1000):
    wins = 0
    losses = 0
    ties = 0

    for _ in range(num_games):
        print(f'Playing Game: {_}')
        board = np.zeros((BOARD_ROWS, BOARD_COLS))  # Reset board
        game_over = False
        player = 1  # Player 1 starts (AI controlled by model)
        
        # Initialize state before the game starts
        state = state_to_dqn_input(board.flatten())
        
        while not game_over:
            # Model selects the best move
            with torch.no_grad():
                q_values = model(state)  # Get Q-values for each action
            q_values = q_values.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

            # Select the action with the highest Q-value
            valid_actions = np.where(board.flatten() == 0)[0]  # Find all empty spots (valid actions)
            valid_q_values = q_values[valid_actions]  # Get Q-values for only valid actions
            action = valid_actions[np.argmax(valid_q_values)]  # Select the valid action with the highest Q-value

            row, col = divmod(action, 3)

            # Make the move
            if board[row][col] == 0:  # Check if the spot is empty
                board[row][col] = player

                # Calculate the reward and check if the player has won
                reward = 0
                if check_win(player, board):
                    reward = 1  # Winning reward
                    if player == 1:
                        wins += 1  # Player 1 wins
                    else:
                        losses += 1  # Player 2 wins
                    game_over = True
                elif is_board_full(board):
                    reward = 0.5  # Tie reward
                    ties += 1  # Tie condition (board full with no winner)
                    game_over = True
                else:
                    reward = 0  # No reward if the game is still ongoing

                # Store the experience in the replay memory
                next_state = state_to_dqn_input(board.flatten())  # Convert the next board state
                done = 1 if game_over else 0  # Done flag (1 if the game is over, else 0)
                replay_memory.push((state, action, reward, next_state, done))  # Add experience to replay memory

                # Update the state for the next move
                state = next_state

                # Switch to the other player
                player = 2 if player == 1 else 1

    return wins, losses, ties

def train_model_DQL(model, optimizer, criterion, replay_memory, batch_size, num_epochs=1000):
    for epoch in range(num_epochs):
        # Ensure replay memory has enough experiences before training
        if len(replay_memory) < batch_size:
            continue
        
        model.train_model(optimizer, criterion, replay_memory, batch_size)

        # Periodically evaluate the model during training
        if epoch % 100 == 0:  # Evaluate every 100 epochs
            evaluate_model(model, test_data_file="trainTTT.data")

        # Decay epsilon after each epoch
        model.decay_epsilon()

    print(f"Training completed after {num_epochs} epochs.")

def evaluate_model(model, test_data_file):
    model.eval()
    X_test, y_test = format_data(test_data_file)
    
    with torch.no_grad():
        predictions = model(X_test)
        predicted_labels = torch.sigmoid(predictions).round()  # Round to get 0 or 1
        correct = (predicted_labels == y_test).float()
        accuracy = correct.sum() / len(correct)
    
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Initialize model, optimizer, and loss function
model = DQL()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # Mean squared error loss for regression

# Initialize Replay Memory (assuming a replay memory class is defined)
replay_memory = ReplayMemory(capacity=10000)

# Play 1000 games to fill the replay memory
wins, losses, ties = play_game(model, replay_memory, num_games=2000)
print(f"Games played: Wins: {wins}, Losses: {losses}, Ties: {ties}")

train_model_DQL(model, optimizer, criterion, replay_memory, batch_size=64, num_epochs=1000)

# # Load data
# X, y = format_data('trainTTT.data')  # Replace with your actual file path

# # Training loop
# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode
    
#     for i in range(len(X)):
#         state = X[i].unsqueeze(0)  # Add batch dimension (1, 27)
#         action = torch.argmax(model(state)).item()  # Select action based on Q-values
#         reward = y[i].item()  # Reward (you could have a separate reward function)
#         next_state = X[i]  # In a more complex setup, you would calculate the next state
#         done = 1 if i == len(X) - 1 else 0  # Set 'done' flag for last sample (or use some logic)

#         # Store experience in replay memory
#         replay_memory.push(state, action, reward, next_state, done)

#         # Sample a batch and perform training
#         if len(replay_memory) >= batch_size:
#             model.train_model(optimizer, criterion, replay_memory, batch_size)  # Correct way to call the method

#     print(f"Epoch {epoch + 1}/{num_epochs} completed")

# Test the AI by having it play itself 1000 times
wins, losses, ties = play_game(model, replay_memory, num_games=1000)
print(f"AI Win rate: {wins/1000*100:.2f}%")
print(f"AI Loss rate: {losses/1000*100:.2f}%")
print(f"AI Tie rate: {ties/1000*100:.2f}%")

# Format the data
# file_path_train = 'trainTTT.data'
# file_path_test = 'testTTT.data'
# train_X, train_Y = format_data(file_path_train)
# test_X, test_Y = format_data(file_path_test)

# draw_lines()
# player = 1
# game_over = False

# # Game loop
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             sys.exit()

#         if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
#             mouseX = event.pos[0] // SQUARE_SIZE
#             mouseY = event.pos[1] // SQUARE_SIZE

#             if available_square(mouseY, mouseX):
#                 mark_square(mouseY, mouseX, player)
#                 if check_win(player):
#                     game_over = True
#                 player = player % 2 + 1

#                 if not game_over:
#                     if best_move():
#                         if check_win(2):
#                             game_over = True
#                         player = player % 2 + 1

#                 if not game_over:
#                     if is_board_full():
#                         game_over = True

#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_r:
#                 restart_game()
#                 game_over = False
#                 player = 1

#     if not game_over:
#         draw_figures()
#     else:
#         if check_win(1):
#             draw_figures(GREEN)
#             draw_figures(GREEN)
#         elif check_win(2):
#             draw_figures(RED)
#             draw_figures(RED)
#         else:
#             draw_figures(GRAY)
#             draw_lines(GRAY)

#     pygame.display.update()
