import torch.nn as nn

class CmdNetwork(nn.Module):
    
    def __init__(self, category_number):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 1, 64, T] → [B, 32, 64, T]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # [B, 32, 32, T//2]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # [B, 64, 16, T//4]

            nn.Dropout(0.3)
        )
    
        self.pool = nn.AdaptiveAvgPool2d((16, None))
        self.rnn = nn.GRU(
            input_size=16 * 64,  # 展平 CNN 输出
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, category_number)
        )
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, 64, T]
        x = self.cnn(x)
        x = self.pool(x)  # [B, 64, 16, T']
        b, c, h, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T', C, H]
        x = x.view(b, t, -1)  # [B, T', Feature]
        output, _ = self.rnn(x)
        out = output[:, -1, :]  # 取最后时间步
        return self.classifier(out)
