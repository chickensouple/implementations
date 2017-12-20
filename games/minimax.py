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
		# at each node of tree, the tuple of
		# (game state, minimax value, opt_action, prev action) is saved
		self.tree = Tree()

	def search(self, game_state, player):
		self.reset()
		# create root node
		root_idx = self.tree.insert_node((game_state, None, None, None), None)

		self.count = 0 # used for counting the number of recursive function calls
		val = self.__search_helper(root_idx, player, level=0)
		return val

	def __search_helper(self, node_idx, player, level):
		self.count += 1
		if self.count % 10000 == 0:
			print("tree size: " + str(len(self.tree.node_info)))

		# get next actions
		actions = self.game.get_valid_actions()
		game_state, _, _, prev_action = self.tree.get_node_info(node_idx)

		# if we have reached a leaf node
		if len(actions) == 0:
			winner = self.game.get_winner()
			if player == winner:
				val = 1 # winner
			elif winner == -1:
				val = 0 # tie
			else:
				val = -1

			# update minimax value in the tree
			self.tree.update_node_info(node_idx, (game_state, val, None, prev_action))
			return val

		# recursively search each child node
		minimax_vals = []
		for action in actions:
			self.game.set_state(game_state)
			self.game.step(action)

			new_node_idx = self.tree.insert_node((self.game.get_state(), None, None, action), node_idx)
			val = self.__search_helper(new_node_idx, player, level+1)
			minimax_vals.append(val)

		# take either the minimum or maximum of the child node values
		# depending on whether you are the min player or max player
		self.game.set_state(game_state)
		if player == self.game.get_curr_player():
			val_idx = np.argmax(minimax_vals)
		else:
			val_idx = np.argmin(minimax_vals)
		val = minimax_vals[val_idx]
		opt_action = actions[val_idx]

		self.tree.update_node_info(node_idx, (game_state, val, opt_action, prev_action))
		return val


if __name__ == '__main__':
	from tictactoe import TicTacToe
	import pickle
	import argparse

	parser = argparse.ArgumentParser(description='Minimax TicTacToe Player')
	parser.add_argument('--type', dest='type', type=str, 
		choices=['train', 'test'],
		required=True,
		help='train (build minimax tree) or test')
	parser.add_argument('--file', dest='file', type=str,
		default='ttt.p',
		help='file to save/load minimax tree in')
	parser.add_argument('--player', dest='player', type=int,
		choices=[0, 1],
		default=0,
		help='which player to solve/test minimax tree for')
	args = parser.parse_args()

	player = args.player

	ttt = TicTacToe()
	if args.type == 'train':
		initial_game_state = ttt.get_state()

		mm = MiniMaxSearch(ttt)
		val = mm.search(initial_game_state, player)
		print("Search Ended")
		print("val: " + str(val))
		print("tree size: " + str(len(mm.tree.node_info)))

		# clean tree for saving. only save the necessary information
		for i in range(len(mm.tree.node_info)):
			_, _, opt_action, prev_action = mm.tree.node_info[i]
			mm.tree.node_info[i] = (opt_action, prev_action)

		pickle.dump(mm, open(args.file, 'wb'))
	elif args.type == 'test':
		mm = pickle.load(open(args.file, 'rb'))
		curr_idx = 0

		def get_next_idx(tree, curr_idx, action):
			children_indices = tree.p_c_edges[curr_idx]
			for idx in children_indices:
				_, idx_action = tree.get_node_info(idx)
				if action == idx_action:
					return idx
			raise Exception('No such action')


		while ttt.get_winner() == None:
			if player == 0:
				opt_action, _ = mm.tree.get_node_info(curr_idx)
				ttt.step(opt_action)
				curr_idx = get_next_idx(mm.tree, curr_idx, opt_action)
				ttt.print_board()

				if ttt.get_winner() != None:
					break

				while True:
					try:
						move = raw_input('Enter your move or (q)uit: ')
						if move == 'q':
							exit()
						move = move.split(',')
						x = int(move[0])
						y = int(move[1])
						action = (x, y)
						ttt.step(action)
						curr_idx = get_next_idx(mm.tree, curr_idx, action)
					except:
						continue
					break

			elif player == 1:
				ttt.print_board()
				while True:
					try:
						move = raw_input('Enter your move or (q)uit: ')
						if move == 'q':
							exit()
						move = move.split(',')
						x = int(move[0])
						y = int(move[1])
						action = (x, y)
						ttt.step(action)
						curr_idx = get_next_idx(mm.tree, curr_idx, action)
					except Exception as e:
						print(e)
						continue
					break

				if ttt.get_winner() != None:
					break

				opt_action, _ = mm.tree.get_node_info(curr_idx)
				ttt.step(opt_action)
				curr_idx = get_next_idx(mm.tree, curr_idx, opt_action)
				ttt.print_board()

		ttt.print_board()
		winner = ttt.get_winner()
		if winner == 1 - args.player:
			print("You Lose!")
		elif winner == args.player:
			print("You Win!")
		else:
			print("Tie Game!")

