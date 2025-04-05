import numpy as np
import math
from game import GomokuGame

class TreeNode:
    """MCTS的树节点"""
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent  # 父节点
        self.children = {}  # 子节点，键为动作，值为TreeNode
        self.n_visits = 0  # 访问次数
        self.Q = 0  # 动作价值
        self.U = 0  # UCB值的一部分
        self.P = prior_p  # 先验概率
    
    def expand(self, action_priors):
        """
        扩展节点
        
        参数:
            action_priors: 动作先验概率的列表，每个元素为(action, prior_p)
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(parent=self, prior_p=prob)
    
    def select(self, c_puct):
        """
        选择最有价值的子节点
        
        参数:
            c_puct: 控制探索程度的常数
            
        返回:
            (action, next_node): 选择的动作和对应的子节点
        """
        # 选择具有最大UCB值的子节点
        return max(self.children.items(),
                   key=lambda item: item[1].get_value(c_puct))
    
    def update(self, leaf_value):
        """
        更新节点值
        
        参数:
            leaf_value: 叶节点的评估值
        """
        # 更新访问次数
        self.n_visits += 1
        # 更新Q值，使用增量平均
        self.Q += (leaf_value - self.Q) / self.n_visits
    
    def update_recursive(self, leaf_value):
        """
        递归更新所有祖先节点
        
        参数:
            leaf_value: 叶节点的评估值
        """
        # 如果不是根节点，则先更新父节点
        if self.parent:
            self.parent.update_recursive(-leaf_value)  # 对手的收益是当前玩家的负收益
        # 更新自身
        self.update(leaf_value)
    
    def get_value(self, c_puct):
        """
        计算UCB值
        
        参数:
            c_puct: 控制探索程度的常数
            
        返回:
            UCB值
        """
        # U = c_puct * P * sqrt(parent_n) / (1 + n)
        self.U = c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.U
    
    def is_leaf(self):
        """判断是否为叶节点"""
        return len(self.children) == 0
    
    def is_root(self):
        """判断是否为根节点"""
        return self.parent is None

class MCTS:
    """蒙特卡洛树搜索"""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000):
        """
        初始化
        
        参数:
            policy_value_fn: 策略价值函数，输入棋盘状态，输出(action_probs, value)
            c_puct: 控制探索程度的常数
            n_playout: 每次移动的模拟次数
        """
        self.root = TreeNode(None, 1.0)
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
    
    def playout(self, game):
        """
        从根节点到叶节点进行一次模拟
        
        参数:
            game: 游戏环境
        """
        node = self.root
        while True:
            if node.is_leaf():
                break
            
            # 贪婪地选择下一步行动
            action, node = node.select(self.c_puct)
            game.make_move(action)
        
        # 评估叶节点
        action_probs, leaf_value = self.policy_value_fn(game.get_board_state_for_nn())
        
        # 检查游戏是否结束
        end, winner = game.done, game.winner
        if not end:
            # 扩展叶节点
            valid_moves = game.get_valid_moves()
            action_probs = [(move, action_probs[move[0] * game.board_size + move[1]]) 
                           for move in valid_moves]
            node.expand(action_probs)
        else:
            # 游戏结束，根据胜者确定叶节点的值
            if winner == 0:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == game.current_player else -1.0
        
        # 反向传播更新节点值
        node.update_recursive(-leaf_value)
    
    def get_move_probs(self, game, temp=1e-3):
        """
        获取所有可能动作的概率
        
        参数:
            game: 游戏环境
            temp: 温度参数，控制探索程度
            
        返回:
            move_probs: 动作概率的字典，键为动作，值为概率
        """
        for _ in range(self.n_playout):
            game_copy = GomokuGame(board_size=game.board_size)
            game_copy.board = np.copy(game.board)
            game_copy.current_player = game.current_player
            game_copy.done = game.done
            game_copy.winner = game.winner
            game_copy.last_move = game.last_move
            self.playout(game_copy)
        
        # 计算根节点的子节点的访问次数
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        
        # 计算动作概率
        act_probs = self._softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        
        return acts, act_probs
    
    def _softmax(self, x):
        """Softmax函数"""
        probs = np.exp(x - np.max(x))
        return probs / np.sum(probs)
    
    def update_with_move(self, last_move):
        """
        更新根节点，前进到下一个状态
        
        参数:
            last_move: 最后一步动作
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)
    
    def __str__(self):
        """字符串表示"""
        return "MCTS"

class MCTSPlayer:
    """使用MCTS的AI玩家"""
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=False):
        """
        初始化
        
        参数:
            policy_value_function: 策略价值函数
            c_puct: 控制探索程度的常数
            n_playout: 每次移动的模拟次数
            is_selfplay: 是否为自我对弈
        """
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self.is_selfplay = is_selfplay
    
    def set_player_ind(self, player):
        """设置玩家编号"""
        self.player = player
    
    def reset_player(self):
        """重置玩家"""
        self.mcts.update_with_move(-1)
    
    def get_action(self, game, temp=1e-3, return_prob=False):
        """
        获取动作
        
        参数:
            game: 游戏环境
            temp: 温度参数
            return_prob: 是否返回动作概率
            
        返回:
            如果return_prob为True，返回(move, move_probs)
            否则返回move
        """
        # 获取可能的动作和对应的概率
        move_probs = np.zeros(game.board_size * game.board_size)
        acts, probs = self.mcts.get_move_probs(game, temp)
        move_probs[list(map(lambda x: x[0] * game.board_size + x[1], acts))] = probs
        
        if self.is_selfplay:
            # 添加Dirichlet噪声进行探索（自我对弈时）
            move = np.random.choice(
                len(acts),
                p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
            )
            move = acts[move]
            # 更新根节点
            self.mcts.update_with_move(move)
        else:
            # 贪婪地选择概率最高的动作（与人类对弈时）
            move = acts[np.argmax(probs)]
            # 重置根节点
            self.mcts.update_with_move(-1)
        
        if return_prob:
            return move, move_probs
        else:
            return move
    
    def __str__(self):
        """字符串表示"""
        return "MCTS {}".format(self.player)

if __name__ == "__main__":
    # 初始化游戏环境
    board_size = 15
    game = GomokuGame(board_size=board_size)
    
    # 定义策略价值函数
    def policy_value_fn(board_state):
        # 这里使用随机策略和零价值作为示例
        action_probs = np.ones((board_size, board_size)) / (board_size * board_size)
        value = 0
        return action_probs.flatten(), value
    
    # 初始化MCTS玩家
    player = MCTSPlayer(policy_value_function=policy_value_fn, c_puct=5, n_playout=1000, is_selfplay=True)
    
    # 模拟一局游戏
    while not game.done:
        move, move_probs = player.get_action(game, return_prob=True)
        game.make_move(move)
        print(f"Player {game.current_player} made move {move}")
        
    # 打印游戏结果
    if game.winner == 0:
        print("The game is a draw.")
    else:
        print(f"Player {game.winner} wins!") 