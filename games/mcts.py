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
        else:
        	self.p_c_edges[node_idx] = []
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



class MonteCarloTreeSearch(object):
	def __init__(self, game):
		self.game = game
		self.c = 2
		self.reset()

	def reset(self):
		# tree to keep track of minimax search
		# at each node of tree, the tuple of
		# (game state, value, visit frequency) is saved
		self.tree = Tree()

	def search(self, game_state):
		self.reset()
		self.game.set_state(game_state)

		# create root node
		root_idx = self.tree.insert_node((game_state, None, 0), None)
		self.N = 0
		self.simulate()

	def simulate(self, root_idx):
		# tree policy
		tree_idx = root_idx

        # get current player
        game_state, _, _ = self.tree.get_node_info(tree_idx)
        self.game.set_state(game_state)
        curr_player = self.game.get_curr_player()

		child_indices = self.tree.p_c_edges[tree_idx]
		while len(child_indices) != 0:
			val_arr = np.zeros(len(child_indices))
			visits_arr = np.zeros(len(child_indices))
			for idx in child_indices:
				_, val, visits = self.tree.get_node_info(idx)
				val_arr[idx] = val
				visits_arr[idx] = visits

			ucb = val_arr + self.c * np.sqrt(np.log(self.N) / visits_arr)
			tree_idx = np.argmax(ucb)

			child_indices = self.tree.p_c_edges[tree_idx]

		# expand nodes
		game_state, _, _ = self.tree.get_node_info(tree_idx)
		for action in self.game.get_valid_actions():
			self.game.set_state(game_state)
			self.game.step(action)
			self.tree.insert_node((self.game.get_state(), None, 0), tree_idx)

		# choose random node to expand
		tree_idx = np.random.choice(self.tree.p_c_edges[tree_idx])
        self.rollout(tree_idx)


    def rollout(self, tree_idx, curr_player):
		# fast policy rollout
		game_state, _, _ = self.tree.get_node_info(tree_idx)
		self.game.set_state(game_state)
		while self.game.get_winner() == None:
			actions = self.game.get_valid_actions()
			action_idx = np.random.randint(len(actions))
			action = actions[action_idx]
			self.game.step(action)


        winner = self.game.get_winner()
        if winner == curr_player:
            # win
            val = 1
        elif winner == -1:
            # tie
            val = 0
        else:
            #loss
            val = -1
		


if __name__ == '__main__':
	from tictactoe import TicTacToe

	ttt = TicTacToe()
	initial_state = ttt.get_state()
	mcts = MonteCarloTreeSearch(ttt)
	mcts.search(initial_state)


