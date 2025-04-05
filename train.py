import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import random
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


from game import GomokuGame
from model import GomokuModel
from mcts import MCTSPlayer

class Trainer:
    """五子棋AI训练器"""
    def __init__(self, board_size=15, lr=0.001, batch_size=512, buffer_size=10000, 
                 n_games=1000, n_playout=400, c_puct=5, temp=1.0, temp_drop_step=20,
                 checkpoint_freq=100, model_dir='models', games_per_collect=10, use_gpu=True,
                 max_steps=300, max_game_time=86400, num_threads=4, enable_triton=False):
        """
        初始化
        
        参数:
            board_size: 棋盘大小
            lr: 学习率
            batch_size: 批大小
            buffer_size: 经验回放缓冲区大小
            n_games: 训练游戏数
            n_playout: MCTS每步的模拟次数
            c_puct: MCTS的探索常数
            temp: 初始温度
            temp_drop_step: 温度下降步数
            checkpoint_freq: 保存检查点的频率
            model_dir: 模型保存目录
            games_per_collect: 每次收集的游戏数
            use_gpu: 是否使用GPU
            max_steps: 单局游戏的最大步数
            max_game_time: 单局游戏的最大时间（秒），默认设为24小时，实际上相当于无限制
            num_threads: 数据收集的线程数
            enable_triton: 是否启用Triton加速的MCTS
        """
        self.board_size = board_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_games = n_games
        self.n_playout = n_playout
        self.c_puct = c_puct
        self.temp = temp
        self.temp_drop_step = temp_drop_step
        self.checkpoint_freq = checkpoint_freq
        self.model_dir = model_dir
        self.games_per_collect = games_per_collect
        self.use_gpu = use_gpu
        self.max_steps = max_steps
        self.max_game_time = max_game_time
        self.num_threads = num_threads
        self.enable_triton = enable_triton
        
        # 检测GPU是否可用
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建模型目录
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 初始化模型
        self.model = GomokuModel(board_size=board_size)
        self.model.to(self.device)  # 将模型移动到GPU
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # 经验回放缓冲区
        self.data_buffer = deque(maxlen=buffer_size)
        
        # 训练统计
        self.loss_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        
        # 线程锁，用于保护模型在预测时的线程安全
        self.model_lock = threading.RLock()
        
    def policy_value_fn(self, board_state):
        """
        策略价值函数
        
        参数:
            board_state: 棋盘状态
            
        返回:
            action_probs: 动作概率
            value: 状态价值
        """
        # 使用线程锁保护模型预测的线程安全
        with self.model_lock:
            # 预测时使用GPU
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(board_state).unsqueeze(0)  # 添加batch维度
                
                # 如果使用GPU，则将输入移动到GPU
                if self.use_gpu and torch.cuda.is_available():
                    x = x.to(self.device)
                    
                policy_logits, value = self.model(x)
                
                # 将结果移回CPU并转换为numpy
                action_probs = torch.nn.functional.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                value = value.item()
                
            return action_probs, value
    
    def _collect_game_thread(self, game_id, result_queue):
        """
        在线程中收集单局自我对弈数据
        result_queue中存放了play_data,play_data是一局游戏,
        play_data中存放了state_list,mcts_prob_list,winner_z
        state_list中存放了棋盘状态,mcts_prob_list中存放了mcts的概率,winner_z中存放了胜者
        
        
        参数:
            game_id: 游戏ID
            result_queue: 结果队列，用于存储收集到的数据
        """
        # 创建游戏环境
        game = GomokuGame(board_size=self.board_size)
        
        # 创建AI玩家，根据配置选择是否使用Triton加速
        if self.enable_triton:
            try:
                from triton_mcts import TritonMCTSPlayer
                player = TritonMCTSPlayer(self.model, c_puct=self.c_puct, 
                                         n_playout=self.n_playout, is_selfplay=True)
                print(f"游戏 {game_id} 使用Triton加速的MCTS")
            except ImportError:
                print(f"警告: 游戏 {game_id} 无法导入Triton MCTS，回退到标准MCTS")
                player = MCTSPlayer(self.policy_value_fn, c_puct=self.c_puct, 
                                   n_playout=self.n_playout, is_selfplay=True)
        else:
            player = MCTSPlayer(self.policy_value_fn, c_puct=self.c_puct, 
                               n_playout=self.n_playout, is_selfplay=True)
        
        # 收集一局游戏的数据
        play_data = []
        state_list = []
        mcts_prob_list = []
        current_player_list = []
        
        # 记录开始时间和步数
        start_time = time.time()
        step_count = 0
        
        # 开始游戏
        while True:
            # 检查是否超过最大步数
            if step_count >= self.max_steps:
                print(f"游戏 {game_id} 达到最大步数 {self.max_steps}，强制结束")
                # 将当前游戏标记为平局
                game.done = True
                game.winner = 0
                break
            
            # 注释掉时间限制检查
            # if time.time() - start_time > self.max_game_time:
            #     print(f"游戏 {game_id} 超过最大时间 {self.max_game_time}秒，强制结束")
            #     # 将当前游戏标记为平局
            #     game.done = True
            #     game.winner = 0
            #     break
            
            # 保存当前状态
            state_list.append(game.get_board_state_for_nn())
            current_player_list.append(game.current_player)
            
            # 计算温度
            if len(state_list) < self.temp_drop_step:
                temp = self.temp
            else:
                temp = 1e-3  # 几乎贪婪
            
            # 获取动作和动作概率
            move, move_probs = player.get_action(game, temp=temp, return_prob=True)
            
            # 保存MCTS的概率
            mcts_prob_list.append(move_probs)
            
            # 执行动作
            _, _, done, _ = game.make_move(move)
            step_count += 1
            
            # 如果游戏结束
            if done:
                break
        
        # 游戏结束后处理数据
        if len(state_list) > 0:  # 确保有数据
            # 根据胜者确定奖励
            if game.winner == 0:  # 平局
                winner_z = np.zeros(len(state_list))
            else:
                winner_z = np.ones(len(state_list))
                for j in range(len(state_list)):
                    if current_player_list[j] != game.winner:
                        winner_z[j] = -1
            
            # 将数据添加到play_data
            for j in range(len(state_list)):
                play_data.append((state_list[j], mcts_prob_list[j], winner_z[j]))
            
            print(f"游戏 {game_id} 完成，步数: {step_count}，用时: {time.time() - start_time:.1f}秒")
        else:
            print(f"警告: 游戏 {game_id} 没有收集到数据")
        
        # 将结果放入队列
        result_queue.put(play_data)
    
    def collect_selfplay_data(self, n_games=10, max_steps=300, max_game_time=300):
        """
        使用多线程收集自我对弈数据
        
        参数:
            n_games: 自我对弈的游戏数
            max_steps: 单局游戏的最大步数
            max_game_time: 单局游戏的最大时间（秒）
            
        返回:
            play_data: 收集到的数据
        """
        all_play_data = []
        
        # 自动确定线程数
        cpu_count = multiprocessing.cpu_count()
        # 线程数设置为CPU核心数的2倍，但不超过游戏数量
        num_workers = min(cpu_count * 2, n_games)
        
        # 使用线程池并行收集数据
        print(f"使用 {num_workers} 个线程并行收集 {n_games} 局游戏数据...")
        result_queue = queue.Queue()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有游戏任务
            for i in range(n_games):
                executor.submit(self._collect_game_thread, i+1, result_queue)
            
            # 等待所有任务完成并收集结果
            for _ in range(n_games):
                try:
                    game_data = result_queue.get(timeout=86400 + 10)  # 设置为24小时+10秒，实际上相当于无限制
                    all_play_data.extend(game_data)
                except queue.Empty:
                    print("警告: 某些游戏数据收集超时")
        
        return all_play_data
    
    def train_step(self):
        """
        执行一步训练
        
        返回:
            loss: 总损失
            policy_loss: 策略损失
            value_loss: 价值损失
        """
        # 如果数据不足，返回
        if len(self.data_buffer) < self.batch_size:
            return 0, 0, 0
        
        # 从缓冲区中随机采样
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        # 转换为PyTorch张量并移动到GPU
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs_batch = torch.FloatTensor(np.array(mcts_probs_batch)).to(self.device)
        winner_batch = torch.FloatTensor(np.array(winner_batch).reshape(-1, 1)).to(self.device)
        
        # 前向传播
        self.model.train()
        policy_logits, value = self.model(state_batch)
        
        # 计算损失
        value_loss = nn.MSELoss()(value, winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * nn.LogSoftmax(dim=1)(policy_logits), dim=1))
        loss = value_loss + policy_loss
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 返回损失
        return loss.item(), policy_loss.item(), value_loss.item()
    
    def train(self):
        """训练模型"""
        print("开始训练...")
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 记录最佳损失，用于学习率调度
        best_loss = float('inf')
        
        for i in range(self.n_games):
            # 收集自我对弈数据
            print(f"收集第 {i+1}/{self.n_games} 轮自我对弈数据...")
            play_data = self.collect_selfplay_data(
                self.games_per_collect, 
                max_steps=self.max_steps, 
                max_game_time=self.max_game_time
            )
            
            # 扩充数据集（通过旋转和翻转）
            play_data = self.augment_data(play_data)
            
            # 将数据添加到缓冲区
            self.data_buffer.extend(play_data)
            
            # 多次训练
            print(f"训练第 {i+1}/{self.n_games} 轮...")
            epoch_losses = []
            epoch_policy_losses = []
            epoch_value_losses = []
            
            # 根据数据量确定训练次数
            n_train_steps = min(5, len(self.data_buffer) // self.batch_size)
            
            for _ in tqdm(range(n_train_steps)):
                loss, policy_loss, value_loss = self.train_step()
                epoch_losses.append(loss)
                epoch_policy_losses.append(policy_loss)
                epoch_value_losses.append(value_loss)
            
            # 计算平均损失
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0
            avg_value_loss = np.mean(epoch_value_losses) if epoch_value_losses else 0
            
            # 记录损失
            self.loss_history.append(avg_loss)
            self.policy_loss_history.append(avg_policy_loss)
            self.value_loss_history.append(avg_value_loss)
            
            # 更新学习率
            self.scheduler.step(avg_loss)
            
            # 打印进度
            elapsed_time = time.time() - start_time
            print(f"游戏 {i+1}/{self.n_games}, "
                  f"数据缓冲区大小: {len(self.data_buffer)}, "
                  f"损失: {avg_loss:.4f}, "
                  f"策略损失: {avg_policy_loss:.4f}, "
                  f"价值损失: {avg_value_loss:.4f}, "
                  f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}, "
                  f"已用时间: {elapsed_time/60:.1f}分钟")
            
            # 保存检查点
            if (i + 1) % self.checkpoint_freq == 0:
                self.save_checkpoint(f"checkpoint_{i+1}.pt")
                self.plot_loss()
            
            # 更新最佳损失
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_model.pt")
        
        # 保存最终模型
        self.save_checkpoint("final_model.pt")
        self.plot_loss()
        
        # 打印总用时
        total_time = time.time() - start_time
        print(f"训练完成！总用时: {total_time/60:.1f}分钟")
    
    def augment_data(self, play_data):
        """
        通过旋转和翻转扩充数据集
        
        参数:
            play_data: 原始数据
            
        返回:
            augmented_data: 扩充后的数据
        """
        augmented_data = []
        
        for state, mcts_prob, winner in play_data:
            # 原始数据
            augmented_data.append((state, mcts_prob, winner))
            
            # 获取棋盘状态
            board_state = state
            probs = mcts_prob.reshape(self.board_size, self.board_size)
            
            # 旋转和翻转
            for i in range(1, 4):  # 旋转90°, 180°, 270°
                # 旋转棋盘状态
                rot_state = np.array([np.rot90(board_state[0], i),
                                     np.rot90(board_state[1], i),
                                     np.rot90(board_state[2], i)])
                
                # 旋转概率
                rot_probs = np.rot90(probs, i).flatten()
                
                # 添加到扩充数据
                augmented_data.append((rot_state, rot_probs, winner))
                
                # 水平翻转
                flip_state = np.array([np.fliplr(rot_state[0]),
                                      np.fliplr(rot_state[1]),
                                      np.fliplr(rot_state[2])])
                
                # 水平翻转概率
                flip_probs = np.fliplr(np.rot90(probs, i)).flatten()
                
                # 添加到扩充数据
                augmented_data.append((flip_state, flip_probs, winner))
        
        return augmented_data
    
    def save_checkpoint(self, filename):
        """
        保存检查点
        
        参数:
            filename: 文件名
        """
        filepath = os.path.join(self.model_dir, filename)
        
        # 保存模型、优化器和学习率调度器状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_history': self.loss_history,
            'policy_loss_history': self.policy_loss_history,
            'value_loss_history': self.value_loss_history,
        }, filepath)
        
        print(f"模型已保存到 {filepath}")
    
    def load_checkpoint(self, filename):
        """
        加载检查点
        
        参数:
            filename: 文件名
        """
        filepath = os.path.join(self.model_dir, filename)
        
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
            
            if 'policy_loss_history' in checkpoint:
                self.policy_loss_history = checkpoint['policy_loss_history']
            
            if 'value_loss_history' in checkpoint:
                self.value_loss_history = checkpoint['value_loss_history']
            
            print(f"模型已从 {filepath} 加载")
        else:
            print(f"找不到检查点文件: {filepath}")
    
    def plot_loss(self):
        """绘制损失曲线"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.loss_history)
        plt.title('总损失')
        plt.xlabel('训练步数')
        plt.ylabel('损失')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.policy_loss_history)
        plt.title('策略损失')
        plt.xlabel('训练步数')
        plt.ylabel('损失')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.value_loss_history)
        plt.title('价值损失')
        plt.xlabel('训练步数')
        plt.ylabel('损失')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'loss_curve.png'))
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="五子棋AI训练")
    parser.add_argument("--board_size", type=int, default=15, help="棋盘大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=512, help="批大小")
    parser.add_argument("--buffer_size", type=int, default=10000, help="经验回放缓冲区大小")
    parser.add_argument("--n_games", type=int, default=1000, help="训练游戏数")
    parser.add_argument("--n_playout", type=int, default=400, help="MCTS每步的模拟次数")
    parser.add_argument("--c_puct", type=float, default=5, help="MCTS的探索常数")
    parser.add_argument("--temp", type=float, default=1.0, help="初始温度")
    parser.add_argument("--temp_drop_step", type=int, default=20, help="温度下降步数")
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="保存检查点的频率")
    parser.add_argument("--model_dir", type=str, default="models", help="模型保存目录")
    parser.add_argument("--games_per_collect", type=int, default=10, help="每次收集的游戏数")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU")
    parser.add_argument("--resume", type=str, default="best_model.pt", help="从检查点恢复训练")
    parser.add_argument("--max_steps", type=int, default=300, help="单局游戏的最大步数")
    parser.add_argument("--max_game_time", type=int, default=86400, help="单局游戏的最大时间（秒），设为86400（24小时）表示实际上没有时间限制")
    parser.add_argument("--num_threads", type=int, default=4, help="数据收集的线程数")
    parser.add_argument("--enable_triton", action="store_true", help="是否启用Triton加速的MCTS")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = Trainer(
        board_size=args.board_size,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        n_games=args.n_games,
        n_playout=args.n_playout,
        c_puct=args.c_puct,
        temp=args.temp,
        temp_drop_step=args.temp_drop_step,
        checkpoint_freq=args.checkpoint_freq,
        model_dir=args.model_dir,
        games_per_collect=args.games_per_collect,
        use_gpu=args.use_gpu,
        max_steps=args.max_steps,
        max_game_time=args.max_game_time,
        num_threads=args.num_threads,
        enable_triton=args.enable_triton
    )
    
    # 从检查点恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 训练模型
    trainer.train() 