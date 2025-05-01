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
import Game as TTT
from time import time
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

# Define memory for Experience Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, experience):
        """Store experience in memory."""
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
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
        winner = 0
        round = round + 1
        board = np.zeros((BOARD_ROWS, BOARD_COLS))
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
            if player ==1:
                action = model1.select_action(state)
                bestMove = [(int(action/3)), (action%3), action]
            else:
                action = model2.select_action(state)
                bestMove = [(int(action/3)), (action%3), action]

            # Make the move
            if board[bestMove[0]][bestMove[1]] == 0:
                board[bestMove[0]][bestMove[1]] = player
                # Calculate the reward and check if the player has won
                reward = 0
                
                if check_win(board) != -1:
                    if check_win(board) == 2:
                        reward = 1
                        winner = 2
                        p2wins += 1
                        p1losses += 1
                        game_over = True
                        print("Player 2 wins\n")
                    else:
                        reward = -1
                        winner = 1
                        p1wins +=1
                        p2losses += 1
                        game_over = True
                        print("Player 1 wins\n")
                elif is_board_full(board):
                    reward = 0.5
                    ties += 1 
                    game_over = True
                    print("It's a tie\n")
                else:
                    reward = 0

                # Store the experience in the replay memory
                next_state = DQL.state_to_dqn_input(board.flatten())
                done = 1 if game_over else 0
                if winner == 1:
                    print(reward)
                    replay_memory1.push((state, bestMove[2], (reward * -1), next_state, done))
                    replay_memory2.push((state, bestMove[2], reward, next_state, done))
                elif winner ==2:
                    print(reward)
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


def run_game_loop(model):
    global player, board, game_over

    TTT.draw_lines()
    board = np.zeros((BOARD_ROWS, BOARD_COLS))
    player = 2
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
                player = 2

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and player == 1:
                mouseX = event.pos[0] // SQUARE_SIZE
                mouseY = event.pos[1] // SQUARE_SIZE

                if TTT.available_square(mouseY, mouseX):
                    TTT.mark_square(mouseY, mouseX, 1)
                    board[mouseY][mouseX] = 1

                    if check_win(board) == 1 or is_board_full(board):
                        game_over = True
                    else:
                        player = 2

        # AI Move
        if not game_over and player == 2:
            pygame.time.delay(500)
            state = DQL.state_to_dqn_input(board.flatten())
            action = model.best_action(state, board)
            row = int(action/3)
            col = action%3
            if TTT.available_square(row, col) and board[row][col]==0:
                TTT.mark_square(row, col, 2)
                board[row][col] = 2

                if check_win(board) == 2 or is_board_full(board):
                    game_over = True
                else:
                    player = 1 

        # Draw board and updates
        if not game_over:
            TTT.draw_figures()
        else:
            if check_win(board) == 1:
                TTT.draw_figures(GREEN)
                TTT.draw_lines(GREEN)
            elif check_win(board) == 2:
                TTT.draw_figures(RED)
                TTT.draw_lines(RED)
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
start = time()
model1.trainDataset(optimizer1, criterion)
model2.trainDataset(optimizer2, criterion)
# Initialize Replay Memory (assuming a replay memory class is defined)
replay_memory1 = ReplayMemory(capacity=10000)
replay_memory2 = ReplayMemory(capacity=10000)

# Play 1000 games to fill the replay memory
p1wins, p2wins, p1losses, p2losses, ties = play_game(model1, model2, replay_memory1, replay_memory2, num_games=2000)
print(f"Player1 Games played: Wins: {p1wins}, Losses: {p1losses}, Ties: {ties}")
print(f"Player2 Games played: Wins: {p2wins}, Losses: {p2losses}, Ties: {ties}")
if p1losses > p1wins:
    model2.train_model(optimizer2, criterion, replay_memory2, 64, 25)
    bestModel = model2
else:
    model1.train_model(optimizer1, criterion, replay_memory1, 64, 25)
    bestModel = model1
end = time()
print(end - start)
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

run_game_loop(bestModel)
