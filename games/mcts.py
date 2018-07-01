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
        self.p_c_edges[child_idx] = []
        self.p_c_edges[parent_idx].append(child_idx)

    def get_node_info(self, node_idx):
        return self.node_info[node_idx]

    def update_node_info(self, node_idx, node_info):
        self.node_info[node_idx] = node_info



class MonteCarloTreeSearch(object):
    def __init__(self, game):
        self.game = game
        self.c = 2.0
        self.reset()

    def reset(self):
        # tree to keep track of minimax search
        # at each node of tree, the tuple of
        # (game state, value, visit frequency, action_from) is saved
        self.tree = Tree()

    def search(self, game_state, num_iter=1000):
        self.reset()
        self.game.set_state(game_state)

        # create root node
        root_idx = self.tree.insert_node((game_state, 0, 0, None), None)
        self.N = 0
        for _ in range(num_iter):
            self.simulate(root_idx)

        # pick max action
        children_indices = self.tree.p_c_edges[root_idx]
        max_val = -np.Inf
        max_action = None

        for idx in children_indices:
            _, val, _, action = self.tree.get_node_info(idx)
            if val > max_val:
                max_val = val
                max_action = action

        return max_action, max_val

    def simulate(self, root_idx):
        tree_idx = root_idx

        # get current player
        game_state, _, _, _ = self.tree.get_node_info(tree_idx)
        self.game.set_state(game_state)
        curr_player = self.game.get_curr_player()

        # tree policy
        child_indices = self.tree.p_c_edges[tree_idx]
        while len(child_indices) != 0:
            val_arr = np.zeros(len(child_indices))
            visits_arr = np.zeros(len(child_indices))
            for i, idx in enumerate(child_indices):
                _, val, visits, _ = self.tree.get_node_info(idx)
                val_arr[i] = val
                visits_arr[i] = visits

            ucb = val_arr + self.c * np.sqrt(np.log(self.N) / visits_arr)
            max_idx = np.argmax(ucb)

            tree_idx = self.tree.p_c_edges[tree_idx][max_idx]
            child_indices = self.tree.p_c_edges[tree_idx]

        # expand nodes
        game_state, _, _, _ = self.tree.get_node_info(tree_idx)
        self.game.set_state(game_state)
        for action in self.game.get_valid_actions():
            self.game.set_state(game_state)
            self.game.step(action)
            self.tree.insert_node((self.game.get_state(), 0, 0, action), tree_idx)

        child_indices = self.tree.p_c_edges[tree_idx]
        if len(child_indices) == 0:
            winner = self.game.get_winner()
            if winner == curr_player:
                # win
                val = 1
            elif winner == -1:
                # tie
                val = 0
            else:
                # loss
                val = -1
        else:
            # choose random node to expand
            tree_idx = np.random.choice(child_indices)
            val = self.rollout(tree_idx, curr_player)

        # backprop
        while tree_idx != root_idx:
            old_game_state, old_value, old_visit, old_action = self.tree.get_node_info(tree_idx)

            self.game.set_state(old_game_state)
            if self.game.get_curr_player() != curr_player:
                new_value = (1.0 / (1 + old_visit)) * (old_visit * old_value + val)
            else:
                new_value = (1.0 / (1 + old_visit)) * (old_visit * old_value - val)

            new_visit = old_visit + 1
            self.tree.update_node_info(tree_idx, (old_game_state, new_value, new_visit, old_action))
            tree_idx = self.tree.c_p_edges[tree_idx]
        self.N += 1

    def rollout(self, tree_idx, curr_player):
        # fast policy rollout
        game_state, _, _, _ = self.tree.get_node_info(tree_idx)
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
            # loss
            val = -1

        return val


if __name__ == '__main__':
    from tictactoe import TicTacToe

    # np.random.seed(0)

    ttt = TicTacToe()
    mcts = MonteCarloTreeSearch(ttt)

    while ttt.get_winner() == None:
        ttt.print_board()


        move = raw_input('Enter your move or (q)uit: ')
        if move == 'q':
            exit()
        move = move.split(',')
        x = int(move[0])
        y = int(move[1])
        action = (x, y)
        ttt.step(action)

        if ttt.get_winner() != None:
            break

        # opponent move
        curr_state = ttt.get_state()
        action, val = mcts.search(curr_state, 2000)
        ttt.set_state(curr_state)
        ttt.step(action)

    ttt.print_board()
    winner = ttt.get_winner()
    if winner == 0:
        print("You Lost!")
    elif winner == 1:
        print("You Win!")
    else:
        print("Tie Game!")

