import torch.nn as nn

class CmdNetwork(nn.Module):
    
    def __init__(self, category_number):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # [B, 32, 32, T//2]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))   # [B, 64, 16, T//4]
        )

        self.gru_input_size = 64 * 16  # freq维度变成16，高度flatten
        self.rnn = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, category_number)
        )
        
    def forward(self, x):
        x = self.cnn(x)                  # → [B, 64, 16, T//4]
        x = x.permute(0, 3, 1, 2)        # → [B, T//4, 64, 16]
        x = x.reshape(x.size(0), x.size(1), -1)  # → [B, T//4, 1024]

        out, _ = self.rnn(x)             # → [B, T//4, 256]
        out = out[:, -1, :]              # 取最后时间步（也可以 mean pooling）
        return self.classifier(out)      # → [B, num_classes]
