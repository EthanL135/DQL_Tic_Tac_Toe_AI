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
import DQL as DQL
import Tic_Tac_Toe_AI as TTT

BOARD_ROWS = 3
BOARD_COLS = 3

board = np.zeros((BOARD_ROWS, BOARD_COLS))

# Checks if the game board is full
def is_board_full(check_board=board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if check_board[row][col] == 0:
                return False
    return True

# Checks to see if there's a winner
def check_win(check_board):
    # Check all columns
    for col in range(3):
        if check_board[0][col] == check_board[1][col] == check_board[2][col] != 0:
            return check_board[0][col]

    # Check all rows
    for row in range(3):
        if check_board[row][0] == check_board[row][1] == check_board[row][2] != 0:
            return check_board[row][0]

    # Check diagonals
    if check_board[0][0] == check_board[1][1] == check_board[2][2] !=0:
        return check_board[1][1]

    if check_board[0][2] == check_board[1][1] == check_board[2][0] !=0:
        return check_board[1][1]

    return -1


def best_move(state, player, board):
    # Model selects the best move
    with torch.no_grad():
        if player == 1:
            q_values = model1(state)  # Get Q-values for each action
        else:
            q_values = model2(state)
    q_values = q_values.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

    # Select the action with the highest Q-value
    valid_actions = np.where(board.flatten() == 0)[0]  # Find all empty spots (valid actions)
    valid_q_values = q_values[valid_actions]  # Get Q-values for only valid actions
    action = valid_actions[np.argmax(valid_q_values)]  # Select the valid action with the highest Q-value
    
    # This variable is for marking the correct square in pygame window
    move = (-1, -1)
    row, col = divmod(action, 3)
    move = (row, col)

    if move != (-1, -1):
        return row, col, action
    return row, col, action


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

def play_game(model1, model2, replay_memory1, replay_memory2, num_games=1000):
    p1wins = 0
    p2wins = 0
    p1losses = 0
    p2losses = 0
    ties = 0
    round = 0
    winner = 0
    for _ in range(num_games):
        round = round + 1
        board = np.zeros((BOARD_ROWS, BOARD_COLS))  # Reset board
        game_over = False
        if (round % 2) != 0:
            startingPlayer = 1
            player = 1
        else:
            startingPlayer = 2
            player = 2
        # Initialize state before the game starts
        state = DQL.state_to_dqn_input(board.flatten())
        
        while not game_over:
            bestMove = best_move(state, player, board)

            # Make the move
            if board[bestMove[0]][bestMove[1]] == 0:  # Check if the spot is empty
                board[bestMove[0]][bestMove[1]] = player
                print(board)
                print()
                # Calculate the reward and check if the player has won
                reward = 0
                
                if check_win(board) != -1:
                    if check_win(board) == 2:
                        reward = 1  # Winning reward
                        winner = 2
                        p2wins += 1  # Player 2 wins
                        p1losses += 1
                        game_over = True
                        print("Player 2 Wins!\n\n")
                    else:
                        reward = -1 # Losing reward
                        winner = 1
                        p1wins +=1
                        p2losses += 1  # Player 1 wins
                        game_over = True
                        print("Player 1 Wins!\n\n")
                elif is_board_full(board):
                    reward = 0.5  # Tie reward
                    ties += 1  # Tie condition (board full with no winner)
                    game_over = True
                    print("Its a tie..\n\n")
                else:
                    reward = 0  # No reward if the game is still ongoing

                # Store the experience in the replay memory
                next_state = DQL.state_to_dqn_input(board.flatten())  # Convert the next board state
                done = 1 if game_over else 0  # Done flag (1 if the game is over, else 0)
                if winner == 1:
                    replay_memory1.push((state, bestMove[2], reward, next_state, done))  # Add experience to replay memory
                    replay_memory2.push((state, bestMove[2], (reward * -1), next_state, done))
                elif winner ==2:
                    replay_memory2.push((state, bestMove[2], reward, next_state, done))
                    replay_memory1.push((state, bestMove[2], (reward * -1), next_state, done))
                else:
                    replay_memory1.push((state, bestMove[2], reward, next_state, done))
                    replay_memory2.push((state, bestMove[2], reward, next_state, done))
                # Update the state for the next move
                state = next_state

                # Switch to the other player
                player = 2 if player == 1 else 1

    return p1wins, p2wins, p1losses, p2losses, ties


def run_game_loop():
    global player, board, game_over

    TTT.draw_lines()
    board = np.zeros((BOARD_ROWS, BOARD_COLS))
    player = 1  # Human starts
    game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                TTT.restart_game()
                board = np.zeros((BOARD_ROWS, BOARD_COLS))
                game_over = False
                player = 1  # Reset to human

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and player == 1:
                mouseX = event.pos[0] // SQUARE_SIZE
                mouseY = event.pos[1] // SQUARE_SIZE

                if TTT.available_square(mouseY, mouseX):
                    TTT.mark_square(mouseY, mouseX, 1)
                    board[mouseY][mouseX] = 1

                    if check_win(board) == 1 or is_board_full(board):
                        game_over = True
                    else:
                        player = 2  # Switch to AI

        # AI Move
        if not game_over and player == 2:
            pygame.time.delay(500)  # Optional delay for realism
            state = DQL.state_to_dqn_input(board.flatten())
            row, col, _ = best_move(state, player, board)
            if TTT.available_square(row, col):
                TTT.mark_square(row, col, 2)
                board[row][col] = 2

                if check_win(board) == 2 or is_board_full(board):
                    game_over = True
                else:
                    player = 1  # Switch back to human

        # Draw board and updates
        if not game_over:
            TTT.draw_figures()
        else:
            if check_win(board) == 1:
                TTT.draw_figures(GREEN)
                TTT.draw_figures(GREEN)
            elif check_win(board) == 2:
                TTT.draw_figures(RED)
                TTT.draw_figures(RED)
            else:
                TTT.draw_figures(GRAY)
                TTT.draw_lines(GRAY)

        pygame.display.update()



# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Initialize model, optimizer, and loss function
model1 = DQL.DQL()
model2 = DQL.DQL()
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # Mean squared error loss for regression

# Initialize Replay Memory (assuming a replay memory class is defined)
replay_memory1 = ReplayMemory(capacity=10000)
replay_memory2 = ReplayMemory(capacity=10000)

# Play 1000 games to fill the replay memory
p1wins, p2wins, p1losses, p2losses, ties = play_game(model1, model2, replay_memory1, replay_memory2, num_games=2000)
print(f"Player1 Games played: Wins: {p1wins}, Losses: {p1losses}, Ties: {ties}")
print(f"Player2 Games played: Wins: {p2wins}, Losses: {p2losses}, Ties: {ties}")
if p1losses > p1wins:
    model2.train_model(optimizer2, criterion, replay_memory2, 64, 25)
else:
    model1.train_model(optimizer1, criterion, replay_memory1, 64, 25)

# Initialize pygame
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

run_game_loop()

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
#wins, losses, ties = play_game(model, replay_memory, num_games=1000)
#print(f"AI Win rate: {wins/1000*100:.2f}%")
#print(f"AI Loss rate: {losses/1000*100:.2f}%")
#print(f"AI Tie rate: {ties/1000*100:.2f}%")

# Format the data
# file_path_train = 'trainTTT.data'
# file_path_test = 'testTTT.data'
# train_X, train_Y = format_data(file_path_train)
# test_X, test_Y = format_data(file_path_test)
