import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class GomokuModel(nn.Module):
    """五子棋AI模型"""
    def __init__(self, board_size=15, num_res_blocks=10):
        super(GomokuModel, self).__init__()
        self.board_size = board_size
        
        # 输入层
        self.conv_input = ConvBlock(3, 64)  # 输入通道为3: 黑棋位置、白棋位置、当前玩家
        
        # 残差层
        self.res_blocks = nn.ModuleList([ResBlock(64) for _ in range(num_res_blocks)])
        
        # 策略头（输出动作概率）
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 价值头（输出状态价值）
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入状态，形状为(batch_size, 3, board_size, board_size)
            
        返回:
            policy_logits: 策略输出，形状为(batch_size, board_size * board_size)
            value: 价值输出，形状为(batch_size, 1)
        """
        x = self.conv_input(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy_logits = self.policy_fc(policy)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value
    
    def predict(self, board_state):
        """
        预测给定棋盘状态的动作概率和价值
        
        参数:
            board_state: 棋盘状态，numpy数组，形状为(3, board_size, board_size)
            
        返回:
            action_probs: 动作概率，numpy数组，形状为(board_size * board_size,)
            value: 状态价值，标量
        """
        self.eval()
        device = next(self.parameters()).device  # 获取模型所在设备
        
        with torch.no_grad():
            x = torch.FloatTensor(board_state).unsqueeze(0).to(device)  # 添加batch维度并移动到相应设备
            policy_logits, value = self(x)
            
            # 将logits转换为概率
            action_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
            
        return action_probs, value
    
    def save_checkpoint(self, filepath):
        """保存模型检查点"""
        torch.save({
            'model_state_dict': self.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, filepath):
        """加载模型检查点"""
        device = next(self.parameters()).device  # 获取模型所在设备
        checkpoint = torch.load(filepath, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)

# 测试代码
if __name__ == "__main__":
    # 检测GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = GomokuModel(board_size=15)
    model.to(device)
    
    # 创建随机输入
    x = torch.randn(1, 3, 15, 15).to(device)
    
    # 前向传播
    policy_logits, value = model(x)
    
    print(f"策略输出形状: {policy_logits.shape}")
    print(f"价值输出形状: {value.shape}")
    
    # 测试预测函数
    board_state = np.random.randint(0, 2, size=(3, 15, 15)).astype(np.float32)
    action_probs, value = model.predict(board_state)
    
    print(f"动作概率形状: {action_probs.shape}")
    print(f"价值: {value}") 