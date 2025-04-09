import torch.nn as nn

class CmdNetwork(nn.Module):
    
    def __init__(self, category_number):
        super(CmdNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),  # 卷积层1
            nn.ReLU(),
            nn.MaxPool2d(2),                           # 池化层1 (2x2)
            nn.Conv2d(16, 32, kernel_size=3, stride=1), # 卷积层2
            nn.ReLU(),
            nn.MaxPool2d(2),                           # 池化层2 (2x2)
            nn.AdaptiveAvgPool2d((14, 14)),            # 自适应平均池化，输出固定14x14特征图
            nn.Flatten(),                              # 展平为 [batch, 32*14*14]
            nn.Linear(32 * 14 * 14, 128),              # 全连接层，输入特征数32*14*14=6272
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, category_number)                # 输出层，category_number为类别数
        )
        
    def forward(self, x):
        return self.net(x)
