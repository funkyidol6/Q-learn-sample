import random

class Connect4:
  """Simple Connect4 environment."""
  def __init__(self):
    self.board = [[" " for _ in range(7)] for _ in range(6)]  # 6 rows, 7 columns
    self.current_player = "R"  # Red player starts

  def play(self, col):
    """Places a piece in the chosen column."""
    for row in range(5, 0, -1):  # Check from bottom to top
      if self.board[row][col] == " ":
        self.board[row][col] = self.current_player
        self.current_player = "Y" if self.current_player == "R" else "R"
        return True
    return False  # Column full

  def winner(self):
    """Checks for a winner (either "R" or "Y" or None for draw)."""
    # Check rows
    for row in range(6):
      for col in range(4):
        if (self.board[row][col] == self.board[row][col + 1] == 
            self.board[row][col + 2] == self.board[row][col + 3] != " "):
          return self.board[row][col]

    # Check columns
    for col in range(7):
      for row in range(3):
        if (self.board[row][col] == self.board[row + 1][col] == 
            self.board[row + 2][col] == self.board[row + 3][col] != " "):
          return self.board[row][col]

    # Check diagonals
    for row in range(3):
      for col in range(4):
        if (self.board[row][col] == self.board[row + 1][col + 1] == 
            self.board[row + 2][col + 2] == self.board[row + 3][col + 3] != " "):
          return self.board[row][col]
        if (self.board[row][col + 3] == self.board[row + 1][col + 2] == 
            self.board[row + 2][col + 1] == self.board[row + 3][col] != " "):
          return self.board[row][col + 3]

    # Check for draw (board full)
    for col in range(7):
      if self.board[0][col] == " ":
        return None  # Empty space means game not over

    return "D"  # Draw

  def print_board(self):
    """Prints the current state of the Connect4 board."""
    for row in self.board:
      print("|", end="")
      for cell in row:
        print(cell + "|", end="")
      print()

class QLearningAgent:
  """Simple Q-learning agent for Connect4."""
  def __init__(self, alpha=0.1, gamma=0.9):
    self.name = "QLearning Agent"
    self.alpha = alpha  # Learning rate
    self.gamma = gamma  # Discount factor
    self.Q = {}  # Q-value table (state, action) -> reward

  def get_state(self, env):
    """Converts the current board state to a unique string representation."""
    return ''.join(["".join(row) for row in env.board])

  def choose_action(self, env):
    """Chooses an action based on the Q-value table and exploration."""
    state = self.get_state(env)
    # Explore new options with a small probability
    if random.random() < 0.1:
      return random.randint(0, 6)
    # Choose the action with the highest Q-value for the current state
    valid_actions = [col for col in range(7) if env.play(col)]
    if not valid_actions:
      return None  # No valid moves (board full)
    best_action = valid_actions[0]
    best_value = float('-inf')
    for action in valid_actions:
      q_value = self.Q.get((state, action), 0)  # Default to 0 for unseen states
      if q_value > best_value:
        best_action = action
        best_value = q_value
    return best_action

  def update(self, state, action, reward, next_state):
    """Updates the Q-value table based on experience."""
    q_value = self.Q.get((state, action), 0)
    # Bellman equation for Q-learning update
    max_q_next = max(self.Q.get((next_state, a), 0) for a in range(7))
    new_q_value = q_value + self.alpha * (reward + self.gamma * max_q_next - q_value)
    self.Q[(state, action)] = new_q_value


def play_game(env, agent1, agent2):
  """Plays a game of Connect4 between two agents."""
  while True:
    env.print_board()
    col = agent1.choose_action(env)
    print(f"{agent1.name} plays column {col}")
    winner = env.winner()
    if winner:
      break
    col = agent2.choose_action(env)
    print(f"{agent2.name} plays column {col}")
    winner = env.winner()
    if winner:
      break
  env.print_board()
  if winner == "D":
    print("It's a draw!")
  else:
    print(f"{winner} wins!")

class Agent:
  """Simple agent that randomly chooses a valid move."""
  def __init__(self):
    self.name = "Random Agent"

  def choose_action(self, env):
    """Selects a random valid move (column) on the board."""
    while True:
      col = random.randint(0, 6)
      if env.play(col):
        return col

# Example usage
env = Connect4()
agent1 = QLearningAgent()  # Learning agent
agent2 = Agent()  # Random agent (for now)

# Train the agent by playing multiple games
for _ in range(1):
  play_game(env, agent1, agent2)