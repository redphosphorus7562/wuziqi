import numpy as np
import pygame
import sys
import os
import time
import torch

from game import GomokuGame
from model import GomokuModel
from mcts import MCTSPlayer

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (220, 180, 70)
LINE_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 0, 0)

class GomokuGUI:
    """五子棋图形界面"""
    def __init__(self, board_size=15, cell_size=40, model_path=None, n_playout=1000, use_gpu=True):
        """
        初始化
        
        参数:
            board_size: 棋盘大小
            cell_size: 单元格大小
            model_path: 模型路径
            n_playout: AI每步的模拟次数
            use_gpu: 是否使用GPU
        """
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = 40
        self.width = self.board_size * self.cell_size + 2 * self.margin
        self.height = self.board_size * self.cell_size + 2 * self.margin + 100  # 额外空间用于显示信息
        
        # 初始化游戏
        self.game = GomokuGame(board_size=board_size)
        
        # 检测GPU是否可用
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 初始化AI
        if model_path and os.path.exists(model_path):
            self.model = GomokuModel(board_size=board_size)
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            self.ai = MCTSPlayer(self.model.predict, c_puct=5, n_playout=n_playout)
            self.use_ai = True
            print(f"AI模型已从 {model_path} 加载")
        else:
            self.use_ai = False
            print("未找到AI模型，将使用人类玩家")
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("五子棋")
        self.font = pygame.font.SysFont("simsun", 24)
        self.clock = pygame.time.Clock()
        
        # 游戏状态
        self.human_player = 1  # 人类玩家为黑棋
        self.ai_player = 2     # AI玩家为白棋
        self.game_over = False
        self.winner = None
        self.last_move = None
        
    def draw_board(self):
        """绘制棋盘"""
        # 填充背景
        self.screen.fill(WHITE)
        
        # 绘制棋盘
        board_rect = pygame.Rect(
            self.margin - self.cell_size // 2,
            self.margin - self.cell_size // 2,
            self.board_size * self.cell_size,
            self.board_size * self.cell_size
        )
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect)
        
        # 绘制网格线
        for i in range(self.board_size):
            # 横线
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (self.margin, self.margin + i * self.cell_size),
                (self.margin + (self.board_size - 1) * self.cell_size, self.margin + i * self.cell_size),
                2
            )
            # 竖线
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (self.margin + i * self.cell_size, self.margin),
                (self.margin + i * self.cell_size, self.margin + (self.board_size - 1) * self.cell_size),
                2
            )
        
        # 绘制棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.game.board[i, j] != 0:
                    center = (self.margin + j * self.cell_size, self.margin + i * self.cell_size)
                    color = BLACK if self.game.board[i, j] == 1 else WHITE
                    border_color = WHITE if self.game.board[i, j] == 1 else BLACK
                    
                    # 绘制棋子
                    pygame.draw.circle(self.screen, color, center, self.cell_size // 2 - 2)
                    pygame.draw.circle(self.screen, border_color, center, self.cell_size // 2 - 2, 2)
                    
                    # 高亮最后一步
                    if self.last_move and self.last_move == (i, j):
                        pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, center, self.cell_size // 4, 2)
        
        # 绘制状态信息
        self.draw_status()
    
    def draw_status(self):
        """绘制状态信息"""
        status_y = self.margin + self.board_size * self.cell_size + 20
        
        if self.game_over:
            if self.winner == 0:
                status_text = "游戏结束，平局！"
            elif self.winner == self.human_player:
                status_text = "游戏结束，你赢了！"
            else:
                status_text = "游戏结束，AI赢了！"
        else:
            if self.game.current_player == self.human_player:
                status_text = "轮到你下棋（黑棋）"
            else:
                status_text = "轮到AI下棋（白棋）"
        
        status_surface = self.font.render(status_text, True, TEXT_COLOR)
        self.screen.blit(status_surface, (self.width // 2 - status_surface.get_width() // 2, status_y))
        
        # 绘制重新开始按钮
        restart_y = status_y + 40
        restart_text = "重新开始"
        restart_surface = self.font.render(restart_text, True, TEXT_COLOR)
        restart_rect = restart_surface.get_rect(center=(self.width // 2, restart_y))
        pygame.draw.rect(self.screen, (200, 200, 200), restart_rect.inflate(20, 10))
        pygame.draw.rect(self.screen, BLACK, restart_rect.inflate(20, 10), 2)
        self.screen.blit(restart_surface, restart_rect)
        
        return restart_rect
    
    def get_position_from_mouse(self, pos):
        """从鼠标位置获取棋盘位置"""
        x, y = pos
        
        # 计算行列
        row = round((y - self.margin) / self.cell_size)
        col = round((x - self.margin) / self.cell_size)
        
        # 检查是否在棋盘范围内
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return row, col
        else:
            return None
    
    def human_move(self, position):
        """人类玩家下棋"""
        if position is None or self.game.board[position] != 0:
            return False
        
        # 执行动作
        _, _, done, info = self.game.make_move(position)
        self.last_move = position
        
        # 检查游戏是否结束
        if done:
            self.game_over = True
            self.winner = self.game.winner
        
        return True
    
    def ai_move(self):
        """AI玩家下棋"""
        if self.use_ai and not self.game_over and self.game.current_player == self.ai_player:
            # 显示思考中
            self.draw_thinking()
            
            # 获取AI动作
            start_time = time.time()
            move = self.ai.get_action(self.game)
            end_time = time.time()
            
            print(f"AI思考时间: {end_time - start_time:.2f}秒")
            
            # 执行动作
            _, _, done, info = self.game.make_move(move)
            self.last_move = move
            
            # 检查游戏是否结束
            if done:
                self.game_over = True
                self.winner = self.game.winner
    
    def draw_thinking(self):
        """绘制AI思考中的提示"""
        # 绘制棋盘
        self.draw_board()
        
        # 绘制思考中提示
        thinking_text = "AI思考中..."
        thinking_surface = self.font.render(thinking_text, True, TEXT_COLOR)
        thinking_rect = thinking_surface.get_rect(center=(self.width // 2, self.height - 30))
        pygame.draw.rect(self.screen, WHITE, thinking_rect.inflate(20, 10))
        self.screen.blit(thinking_surface, thinking_rect)
        
        # 更新显示
        pygame.display.flip()
    
    def restart_game(self):
        """重新开始游戏"""
        self.game.reset()
        self.game_over = False
        self.winner = None
        self.last_move = None
        
        # 如果AI是黑棋，则AI先行
        if self.human_player == 2 and self.use_ai:
            self.ai_move()
    
    def run(self):
        """运行游戏"""
        running = True
        restart_rect = None
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 检查是否点击了重新开始按钮
                    if restart_rect and restart_rect.collidepoint(event.pos):
                        self.restart_game()
                    # 人类玩家下棋
                    elif not self.game_over and self.game.current_player == self.human_player:
                        position = self.get_position_from_mouse(event.pos)
                        if position:
                            if self.human_move(position):
                                # 如果人类成功下棋且游戏未结束，则AI下棋
                                if not self.game_over and self.use_ai:
                                    # 稍微延迟，让玩家看清楚
                                    self.draw_board()
                                    pygame.display.flip()
                                    time.sleep(0.5)
                                    self.ai_move()
            
            # 绘制棋盘
            self.draw_board()
            
            # 获取重新开始按钮的矩形
            restart_rect = self.draw_status()
            
            # 更新显示
            pygame.display.flip()
            
            # 控制帧率
            self.clock.tick(30)
        
        pygame.quit()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="五子棋人机对战")
    parser.add_argument("--model_path", type=str, default="models/best_model.pt", help="模型路径")
    parser.add_argument("--board_size", type=int, default=15, help="棋盘大小")
    parser.add_argument("--n_playout", type=int, default=1000, help="AI每步的模拟次数")
    parser.add_argument("--human_player", type=int, default=1, choices=[1, 2], help="人类玩家编号，1为黑棋，2为白棋")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU")
    
    args = parser.parse_args()
    
    # 创建游戏
    gui = GomokuGUI(
        board_size=args.board_size,
        cell_size=40,
        model_path=args.model_path,
        n_playout=args.n_playout,
        use_gpu=args.use_gpu
    )
    
    # 设置人类玩家
    gui.human_player = args.human_player
    gui.ai_player = 3 - args.human_player  # 1->2, 2->1
    
    # 如果AI是黑棋，则AI先行
    if gui.human_player == 2 and gui.use_ai:
        gui.ai_move()
    
    # 运行游戏
    gui.run() 