import numpy as np

class Tree(object):
    def __init__(self):
        # node_info is extra info about the state
        # for examples, costs, cached values, paths, etc.
        self.node_info = []

        # maps a child_idx to a parent_idx
        self.c_p_edges = dict()

        # maps a parent_idx to a list of child_idx
        self.p_c_edges = dict()

    def clear(self):
        self.node_info = []
        self.c_p_edges = dict()
        self.p_c_edges = dict()

    def insert_node(self, node_info, parent_idx):
    	node_idx = len(self.node_info)
    	self.node_info.append(node_info)

        if parent_idx != None:
            self.insert_edge(node_idx, parent_idx)
        return node_idx

    def insert_edge(self, child_idx, parent_idx):
        self.c_p_edges[child_idx] = parent_idx
        if parent_idx in self.p_c_edges:
            self.p_c_edges[parent_idx].append(child_idx)
        else:
            self.p_c_edges[parent_idx] = [child_idx]

    def get_node_info(self, node_idx):
    	return self.node_info[node_idx]

    def update_node_info(self, node_idx, node_info):
    	self.node_info[node_idx] = node_info

class MiniMaxSearch(object):
	def __init__(self, game):
		self.game = game
		self.reset()

	def reset(self):
		# tree to keep track of minimax search
		# at each node of tree, the game state is saved
		self.tree = Tree()

	def search(self, game_state):
		self.reset()
		# create root node
		root_idx = self.tree.insert_node((game_state, None), None)
		self.game.set_state(game_state)
		player = self.game.get_curr_player()

		self.count = 0
		val = self.__search_helper(root_idx, player, level=0)
		return val

	def __search_helper(self, node_idx, player, level):
		self.count += 1
		if self.count % 1000 == 0:
			print("tree len: " + str(len(self.tree.node_info)))

		actions = self.game.get_valid_actions()
		game_state, _ = self.tree.get_node_info(node_idx)

		if len(actions) == 0:
			winner = self.game.get_winner()
			if player == winner:
				val = 1 # winner
			elif winner == -1:
				val = 0.5 # tie
			else:
				val = 0

			self.tree.update_node_info(node_idx, (game_state, val))

			return val

		minimax_vals = []
		for action in actions:
			self.game.set_state(game_state)
			self.game.make_action(action)

			new_node_idx = self.tree.insert_node((self.game.get_state(), None), node_idx)
			val = self.__search_helper(new_node_idx, player, level+1)
			minimax_vals.append(val)

		self.game.set_state(game_state)
		if player == self.game.get_curr_player():
			val = np.max(minimax_vals)
		else:
			val = np.min(minimax_vals)
		self.tree.update_node_info(node_idx, (game_state, val))

		return val


if __name__ == '__main__':
	from tictactoe import TicTacToe
	ttt = TicTacToe()

	for i in range(4):
		valid_actions = ttt.get_valid_actions()
		action = valid_actions[0]
		ttt.make_action(action)
	ttt.print_board()


	initial_game_state = ttt.get_state()

	mm = MiniMaxSearch(ttt)
	val = mm.search(initial_game_state)
	print("Search Ended")
	print("val: " + str(val))
	print("tree size: " + str(len(mm.tree.node_info)))


