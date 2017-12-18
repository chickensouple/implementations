import numpy as np

class TicTacToe(object):
	"""
	Simple Tic Tac Toe game
	
	Attributes:
	    board (np array): (3, 3) array representing board state
	    -1 is empty, 0 is player 0, 1, is player 1
	    curr_player (int): current player (0 or 1).
	    winner (int): indicates which player is the winner,
	    or -1 if the game is a tie
	    If game has not ended, this will be None
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		"""
		Resets game state to start of a new game
		"""
		self.board = np.ones((3, 3), dtype=np.int8) * -1
		self.curr_player = 0 # 0 or 1
		self.winner = None # None, 0, 1

	def get_valid_actions(self):
		"""
		Get current valid actions for the current player
		
		Returns:
		    list: list of (x, y) tuples of game positions that can be played
		"""
		if self.winner != None:
			return []

		x_vals, y_vals = np.where(self.board == -1)
		valid_actions = []
		for x, y in zip(x_vals, y_vals):
			valid_actions.append((x, y))
		return valid_actions

	def get_curr_player(self):
		"""
		Returns 0 or 1 to indicate which player the current action is for
		"""
		return self.curr_player

	def get_winner(self):
		"""
		Returns 0 or 1 indicating which player won.
		-1 if the game is a tie
		If game has not ended, return None
		"""
		return self.winner

	def get_state(self):
		board_copy = np.copy(self.board)
		return (board_copy, self.curr_player, self.winner)

	def set_state(self, state):
		self.board = np.copy(state[0])
		self.curr_player = state[1]
		self.winner = state[2]

	def step(self, action):
		"""
		takes an action for the current player
		
		Args:
		    action (tuple): (x, y) location of action on the board	
		"""
		if self.winner != None:
			return

		if self.board[action] != -1:
			self.print_board()
			raise Exception('Invalid Move')

		self.board[action] = self.curr_player
		self.curr_player = 1 - self.curr_player
		self.winner = self._check_winner()

	def print_board(self):
		"""
		Prints current state of game board
		player 0 is indicated by 'O'
		player 1 is indicated by 'X'
		"""
		print('-------------')

		print_dict = {-1: ' ', 0: 'O', 1: 'X'}
		for i in range(3):
			print_str = '| '
			for j in range(3):
				print_str += print_dict[self.board[i, j]]
				print_str += ' | '
			print(print_str)
			print('-------------')

	def _check_winner(self):
		for i in range(3):
			# check rows
			row = self.board[i, :]
			if np.all(row == 0):
				return 0
			if np.all(row == 1):
				return 1

			# check columns
			col = self.board[:, i]
			if np.all(col == 0):
				return 0
			if np.all(col == 1):
				return 1

		# check diagonals
		diag = np.diag(self.board) 
		if np.all(diag == 0):
			return 0
		if np.all(diag == 1):
			return 1

		diag = np.diag(np.fliplr(self.board))
		if np.all(diag == 0):
			return 0
		if np.all(diag == 1):
			return 1

		# check for tie
		if np.all(self.board != -1):
			return -1

		return None


if __name__ == '__main__':
	tictactoe = TicTacToe()
	for i in range(20):
		valid_actions = tictactoe.get_valid_actions()

		if len(valid_actions) == 0:
			continue
		action = valid_actions[0]
		tictactoe.step(action)

		tictactoe.print_board()


