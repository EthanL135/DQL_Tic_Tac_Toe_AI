import sys
from collections import deque
import pygame
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import gymnasium as gym

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

# Minmax function that maximizes wins/ties and minimizes losses
# This function ensures that the AI cannot lose and the player cannot win, only tie at best
# def minmax(minimax_board, depth, is_maximizing):
#     # Set penalties for win/loss/tie
#     if check_win(2, minimax_board):
#         return float('inf')
#     elif check_win(1, minimax_board):
#         return float('-inf')
#     elif is_board_full(minimax_board):
#         return 0
#
#     # Main minmax code
#     # AI explores all possible actions to choose the best possible next move
#     if is_maximizing:
#         best_score = -1000
#         for row in range(BOARD_ROWS):
#             for col in range(BOARD_COLS):
#                 if minimax_board[row][col] == 0:
#                     minimax_board[row][col] = 2
#                     score = minmax(minimax_board, depth + 1, False)
#                     minimax_board[row][col] = 0
#                     best_score = max(score, best_score)
#         return best_score
#     else:
#         best_score = 1000
#         for row in range(BOARD_ROWS):
#             for col in range(BOARD_COLS):
#                 if minimax_board[row][col] == 0:
#                     minimax_board[row][col] = 1
#                     score = minmax(minimax_board, depth + 1, True)
#                     minimax_board[row][col] = 0
#                     best_score = min(score, best_score)
#         return best_score

# Determines the AI's next best move using the minmax function
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
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)


    def append(self, transition):
        self.memory.append(transition)


    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)


    def __len__(self):
        return len(self.memory)


class TicTacToeAiDQL():
    # Hyperparameters
    learning_rate = 0.001 # learning rate (alpha)
    discount_factor = 0.9 # discount rate (gamma)
    network_sync_rate = 10
    replay_memory_size = 1000 # size of replay memory
    mini_batch_size = 64 # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss() # NN Loss function
    optimizer = None # NN Optimizer. Initialized later

    ACTIONS = list(range(9))  # Actions correspond to placing a mark in one of the 9 board positions


    def train(self, episodes, render=False):
        # Create Tic-Tac-Toe instance
        env = gym.make('TicTacToe-v0') # FIX LATER
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target neural networks
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        # List to keep track of rewards collected per episode. Initialize list to 0's
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        step_count = 0

        for i in range(episodes):
            state = env.reset()[0] # Initialize to state 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()




draw_lines()
player = 1
game_over = False

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            mouseX = event.pos[0] // SQUARE_SIZE
            mouseY = event.pos[1] // SQUARE_SIZE

            if available_square(mouseY, mouseX):
                mark_square(mouseY, mouseX, player)
                if check_win(player):
                    game_over = True
                player = player % 2 + 1

                if not game_over:
                    if best_move():
                        if check_win(2):
                            game_over = True
                        player = player % 2 + 1

                if not game_over:
                    if is_board_full():
                        game_over = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                restart_game()
                game_over = False
                player = 1

    if not game_over:
        draw_figures()
    else:
        if check_win(1):
            draw_figures(GREEN)
            draw_figures(GREEN)
        elif check_win(2):
            draw_figures(RED)
            draw_figures(RED)
        else:
            draw_figures(GRAY)
            draw_lines(GRAY)

    pygame.display.update()
