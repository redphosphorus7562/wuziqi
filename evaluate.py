import numpy as np
import torch
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from game import GomokuGame
from model import GomokuModel
from mcts import MCTSPlayer

class Evaluator:
    """五子棋AI评估器"""
    def __init__(self, board_size=15, n_games=100, n_playout_1=1000, n_playout_2=1000, model_dir='models', use_gpu=True):
        """
        初始化
        
        参数:
            board_size: 棋盘大小
            n_games: 评估游戏数
            n_playout_1: 玩家1的MCTS模拟次数
            n_playout_2: 玩家2的MCTS模拟次数
            model_dir: 模型目录
            use_gpu: 是否使用GPU
        """
        self.board_size = board_size
        self.n_games = n_games
        self.n_playout_1 = n_playout_1
        self.n_playout_2 = n_playout_2
        self.model_dir = model_dir
        
        # 检测GPU是否可用
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 检查模型目录
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def evaluate_models(self, model1_path, model2_path=None):
        """
        评估两个模型的性能
        
        参数:
            model1_path: 模型1的路径
            model2_path: 模型2的路径，如果为None，则使用随机策略
            
        返回:
            win_rate: 模型1的胜率
            draw_rate: 平局率
            loss_rate: 模型1的负率
        """
        # 加载模型1
        model1 = GomokuModel(board_size=self.board_size)
        
        # 加载检查点
        checkpoint1 = torch.load(model1_path, map_location=self.device)
        if 'model_state_dict' in checkpoint1:
            model1.load_state_dict(checkpoint1['model_state_dict'])
        else:
            model1.load_state_dict(checkpoint1)
            
        model1.to(self.device)
        model1.eval()
        
        player1 = MCTSPlayer(model1.predict, c_puct=5, n_playout=self.n_playout_1)
        
        # 加载模型2或使用随机策略
        if model2_path:
            model2 = GomokuModel(board_size=self.board_size)
            
            # 加载检查点
            checkpoint2 = torch.load(model2_path, map_location=self.device)
            if 'model_state_dict' in checkpoint2:
                model2.load_state_dict(checkpoint2['model_state_dict'])
            else:
                model2.load_state_dict(checkpoint2)
                
            model2.to(self.device)
            model2.eval()
            
            player2 = MCTSPlayer(model2.predict, c_puct=5, n_playout=self.n_playout_2)
        else:
            # 随机策略
            def random_policy(board_state):
                # 随机生成动作概率
                action_probs = np.ones(self.board_size * self.board_size) / (self.board_size * self.board_size)
                return action_probs, 0.0
            
            player2 = MCTSPlayer(random_policy, c_puct=5, n_playout=self.n_playout_2)
        
        # 统计胜负
        win_count = 0
        loss_count = 0
        draw_count = 0
        
        # 记录每局游戏的步数
        game_steps = []
        
        print(f"开始评估，共{self.n_games}局游戏...")
        start_time = time.time()
        
        for i in tqdm(range(self.n_games)):
            # 创建游戏环境
            game = GomokuGame(board_size=self.board_size)
            
            # 设置玩家
            player1.set_player_ind(1)  # 黑棋
            player2.set_player_ind(2)  # 白棋
            
            # 重置玩家
            player1.reset_player()
            player2.reset_player()
            
            # 记录步数
            steps = 0
            
            # 开始游戏
            while True:
                # 黑棋行动
                move = player1.get_action(game)
                _, _, done, _ = game.make_move(move)
                steps += 1
                
                # 检查游戏是否结束
                if done:
                    if game.winner == 1:  # 黑棋胜
                        win_count += 1
                    elif game.winner == 2:  # 白棋胜
                        loss_count += 1
                    else:  # 平局
                        draw_count += 1
                    break
                
                # 白棋行动
                move = player2.get_action(game)
                _, _, done, _ = game.make_move(move)
                steps += 1
                
                # 检查游戏是否结束
                if done:
                    if game.winner == 1:  # 黑棋胜
                        win_count += 1
                    elif game.winner == 2:  # 白棋胜
                        loss_count += 1
                    else:  # 平局
                        draw_count += 1
                    break
            
            # 记录步数
            game_steps.append(steps)
        
        # 计算胜率
        win_rate = win_count / self.n_games
        draw_rate = draw_count / self.n_games
        loss_rate = loss_count / self.n_games
        
        # 计算平均步数
        avg_steps = sum(game_steps) / len(game_steps)
        
        # 计算总用时
        total_time = time.time() - start_time
        
        print(f"评估结果:")
        print(f"胜率: {win_rate:.2%}")
        print(f"平局率: {draw_rate:.2%}")
        print(f"负率: {loss_rate:.2%}")
        print(f"平均步数: {avg_steps:.2f}")
        print(f"总用时: {total_time:.2f}秒，平均每局: {total_time/self.n_games:.2f}秒")
        
        # 绘制胜负图
        self.plot_results(win_rate, draw_rate, loss_rate, avg_steps)
        
        return win_rate, draw_rate, loss_rate
    
    def evaluate_checkpoints(self, checkpoint_pattern="checkpoint_{}.pt", start=100, end=1000, step=100):
        """
        评估多个检查点的性能
        
        参数:
            checkpoint_pattern: 检查点文件名模式
            start: 起始检查点
            end: 结束检查点
            step: 检查点间隔
            
        返回:
            results: 评估结果列表，每个元素为(checkpoint_id, win_rate, draw_rate, loss_rate)
        """
        results = []
        
        # 获取最终模型路径
        final_model_path = os.path.join(self.model_dir, "final_model.pt")
        
        for checkpoint_id in range(start, end + 1, step):
            checkpoint_path = os.path.join(self.model_dir, checkpoint_pattern.format(checkpoint_id))
            
            if os.path.exists(checkpoint_path):
                print(f"评估检查点 {checkpoint_id}...")
                win_rate, draw_rate, loss_rate = self.evaluate_models(checkpoint_path, final_model_path)
                results.append((checkpoint_id, win_rate, draw_rate, loss_rate))
        
        # 绘制学习曲线
        self.plot_learning_curve(results)
        
        return results
    
    def plot_results(self, win_rate, draw_rate, loss_rate, avg_steps):
        """
        绘制评估结果
        
        参数:
            win_rate: 胜率
            draw_rate: 平局率
            loss_rate: 负率
            avg_steps: 平均步数
        """
        plt.figure(figsize=(10, 5))
        
        # 绘制饼图
        plt.subplot(1, 2, 1)
        labels = ['胜', '平', '负']
        sizes = [win_rate, draw_rate, loss_rate]
        colors = ['#66b3ff', '#99ff99', '#ff9999']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('胜负比例')
        
        # 绘制步数直方图
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, f'平均步数: {avg_steps:.2f}', 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=15)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'evaluation_results.png'))
        plt.close()
    
    def plot_learning_curve(self, results):
        """
        绘制学习曲线
        
        参数:
            results: 评估结果列表，每个元素为(checkpoint_id, win_rate, draw_rate, loss_rate)
        """
        if not results:
            return
        
        checkpoint_ids = [r[0] for r in results]
        win_rates = [r[1] for r in results]
        draw_rates = [r[2] for r in results]
        loss_rates = [r[3] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(checkpoint_ids, win_rates, 'o-', label='胜率')
        plt.plot(checkpoint_ids, draw_rates, 's-', label='平局率')
        plt.plot(checkpoint_ids, loss_rates, '^-', label='负率')
        plt.xlabel('检查点')
        plt.ylabel('比率')
        plt.title('学习曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model_dir, 'learning_curve.png'))
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="五子棋AI评估")
    parser.add_argument("--model1_path", type=str, default="models/final_model.pt", help="模型1路径")
    parser.add_argument("--model2_path", type=str, default=None, help="模型2路径，如果不指定则使用随机策略")
    parser.add_argument("--board_size", type=int, default=15, help="棋盘大小")
    parser.add_argument("--n_games", type=int, default=100, help="评估游戏数")
    parser.add_argument("--n_playout_1", type=int, default=1000, help="玩家1的MCTS模拟次数")
    parser.add_argument("--n_playout_2", type=int, default=1000, help="玩家2的MCTS模拟次数")
    parser.add_argument("--evaluate_checkpoints", action="store_true", help="是否评估多个检查点")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = Evaluator(
        board_size=args.board_size,
        n_games=args.n_games,
        n_playout_1=args.n_playout_1,
        n_playout_2=args.n_playout_2,
        model_dir="models",
        use_gpu=args.use_gpu
    )
    
    if args.evaluate_checkpoints:
        # 评估多个检查点
        evaluator.evaluate_checkpoints()
    else:
        # 评估单个模型
        evaluator.evaluate_models(args.model1_path, args.model2_path) 