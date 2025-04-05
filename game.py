import numpy as np

class GomokuGame:
    """五子棋游戏环境"""
    
    def __init__(self, board_size=15):
        """
        初始化游戏
        
        参数:
            board_size: 棋盘大小，默认为15x15
        """
        self.board_size = board_size
        self.reset()
        
    def reset(self):
        """重置游戏状态"""
        # 棋盘状态: 0表示空，1表示黑棋，2表示白棋
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        # 当前玩家: 1表示黑棋，2表示白棋，黑棋先行
        self.current_player = 1
        # 游戏是否结束
        self.done = False
        # 赢家: 0表示没有赢家，1表示黑棋赢，2表示白棋赢，3表示平局
        self.winner = 0
        # 最后一步的位置
        self.last_move = None
        # 历史动作
        self.history = []
        return self.get_state()
    
    def copy(self):
        """创建游戏的深拷贝"""
        game_copy = GomokuGame(board_size=self.board_size)
        game_copy.board = np.copy(self.board)
        game_copy.current_player = self.current_player
        game_copy.done = self.done
        game_copy.winner = self.winner
        game_copy.last_move = self.last_move
        game_copy.history = self.history.copy() if self.history else []
        return game_copy
    
    def get_state(self):
        """获取当前游戏状态"""
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'done': self.done,
            'winner': self.winner,
            'last_move': self.last_move
        }
    
    def get_valid_moves(self):
        """获取所有合法的动作，即所有空位置"""
        if self.done:
            return []
        
        # 所有值为0的位置都是合法动作
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    valid_moves.append((i, j))
        return valid_moves
    
    def make_move(self, position):
        """
        在指定位置落子
        
        参数:
            position: 落子位置，格式为(row, col)
            
        返回:
            state: 新的游戏状态
            reward: 奖励值
            done: 游戏是否结束
            info: 额外信息
        """
        if self.done:
            return self.get_state(), 0, True, {"info": "游戏已结束"}
        
        row, col = position
        
        # 检查位置是否合法
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return self.get_state(), -10, False, {"info": "位置超出棋盘范围"}
        
        # 检查位置是否已经有棋子
        if self.board[row, col] != 0:
            return self.get_state(), -10, False, {"info": "该位置已有棋子"}
        
        # 落子
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.history.append((row, col))
        
        # 检查是否获胜
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1.0 if self.current_player == 1 else -1.0
            return self.get_state(), reward, True, {"info": f"玩家{self.current_player}获胜"}
        
        # 检查是否平局
        if len(self.get_valid_moves()) == 0:
            self.done = True
            self.winner = 3  # 平局
            return self.get_state(), 0, True, {"info": "平局"}
        
        # 切换玩家
        self.current_player = 3 - self.current_player  # 1->2, 2->1
        
        return self.get_state(), 0, False, {"info": "继续游戏"}
    
    def _check_win(self, row, col):
        """
        检查最后一步是否导致获胜
        
        参数:
            row: 最后一步的行
            col: 最后一步的列
            
        返回:
            bool: 是否获胜
        """
        player = self.board[row, col]
        
        # 检查水平方向
        count = 1
        # 向左检查
        for i in range(1, 5):
            if col - i < 0 or self.board[row, col - i] != player:
                break
            count += 1
        # 向右检查
        for i in range(1, 5):
            if col + i >= self.board_size or self.board[row, col + i] != player:
                break
            count += 1
        if count >= 5:
            return True
        
        # 检查垂直方向
        count = 1
        # 向上检查
        for i in range(1, 5):
            if row - i < 0 or self.board[row - i, col] != player:
                break
            count += 1
        # 向下检查
        for i in range(1, 5):
            if row + i >= self.board_size or self.board[row + i, col] != player:
                break
            count += 1
        if count >= 5:
            return True
        
        # 检查左上-右下对角线
        count = 1
        # 向左上检查
        for i in range(1, 5):
            if row - i < 0 or col - i < 0 or self.board[row - i, col - i] != player:
                break
            count += 1
        # 向右下检查
        for i in range(1, 5):
            if row + i >= self.board_size or col + i >= self.board_size or self.board[row + i, col + i] != player:
                break
            count += 1
        if count >= 5:
            return True
        
        # 检查右上-左下对角线
        count = 1
        # 向右上检查
        for i in range(1, 5):
            if row - i < 0 or col + i >= self.board_size or self.board[row - i, col + i] != player:
                break
            count += 1
        # 向左下检查
        for i in range(1, 5):
            if row + i >= self.board_size or col - i < 0 or self.board[row + i, col - i] != player:
                break
            count += 1
        if count >= 5:
            return True
        
        return False
    
    def render(self):
        """打印棋盘状态"""
        symbols = {0: ".", 1: "X", 2: "O"}
        print("  ", end="")
        for i in range(self.board_size):
            print(f"{i:2d}", end="")
        print()
        
        for i in range(self.board_size):
            print(f"{i:2d}", end="")
            for j in range(self.board_size):
                print(f" {symbols[self.board[i, j]]}", end="")
            print()
        
        if self.done:
            if self.winner == 3:
                print("游戏结束，平局！")
            else:
                print(f"游戏结束，玩家{self.winner}获胜！")
        else:
            print(f"当前玩家: {self.current_player}")
            
    def get_board_state_for_nn(self):
        """
        获取适合神经网络输入的棋盘状态
        
        返回:
            numpy数组，形状为(3, board_size, board_size)
            第一个通道表示黑棋的位置
            第二个通道表示白棋的位置
            第三个通道表示当前玩家（全1或全0）
        """
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # 黑棋位置
        state[0] = (self.board == 1).astype(np.float32)
        # 白棋位置
        state[1] = (self.board == 2).astype(np.float32)
        # 当前玩家
        state[2] = np.full((self.board_size, self.board_size), self.current_player == 1, dtype=np.float32)
        
        return state

# 测试代码
if __name__ == "__main__":
    game = GomokuGame(board_size=15)
    game.render()
    
    # 测试几步落子
    moves = [(7, 7), (7, 8), (8, 7), (8, 8), (9, 7), (9, 8), (10, 7), (10, 8), (11, 7)]
    
    for move in moves:
        print(f"\n落子位置: {move}")
        state, reward, done, info = game.make_move(move)
        game.render()
        print(f"信息: {info}")
        
        if done:
            break 