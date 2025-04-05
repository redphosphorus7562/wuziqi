# 五子棋AI训练项目

这个项目旨在训练一个能够下五子棋的AI。项目使用强化学习方法，具体采用了基于策略梯度的方法和蒙特卡洛树搜索(MCTS)。

## 项目结构

- `game.py`: 五子棋游戏环境
- `model.py`: 神经网络模型定义
- `train.py`: 训练脚本
- `mcts.py`: 蒙特卡洛树搜索实现
- `play.py`: 人机对战脚本
- `evaluate.py`: 评估AI性能的脚本

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 训练AI模型：
```bash
python train.py
```

2. 与AI对战：
```bash
python play.py
```

3. 评估AI性能：
```bash
python evaluate.py
``` 