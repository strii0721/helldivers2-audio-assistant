from devtoolkit.Log4P import Log4P
from utils.DatasetUtils import DatasetUtils
from nn.dataset.CmdDataset import CmdDataset
from nn.network.CmdNetworkV2 import CmdNetwork

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


if __name__ == "__main__":
    
    # 前有路径设置的预感
    DATASET_BASE = "src/resources/dat"
    MODEL_PATH = "src/resources/model.pth"
    LOG_FILE_PATH = "src/resources/train.log"
    
    # 前有音频信号设置的预感
    SAMPLE_RATE = 48000
    INTERVAL = 0.8
    
    # 前有神经网络设置的预感
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    INITIAL_LEARNING_RATE = 0.005
    MIN_LEARNING_RATE = 1e-5
    LR_DECREASE_FACTOR = 0.95
    
    logger = Log4P(enable_level = True,
                   enable_timestamp = True,
                   enable_source = True,
                   enable_log_file = False,
                   source = "train")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataframe, category_number = DatasetUtils.get_dataframe_distributed(DATASET_BASE)
    dataset = CmdDataset(dataframe = dataframe,
                         sample_rate = SAMPLE_RATE,
                         length = INTERVAL)
    dataloader = DataLoader(dataset, 
                            batch_size = BATCH_SIZE, 
                            shuffle=True,
                            num_workers = 6,
                            pin_memory=True,
                            persistent_workers=True)
    
    model = CmdNetwork(category_number).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=INITIAL_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 
                                                 gamma = LR_DECREASE_FACTOR)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if inputs.dim() == 3:  # [B, Freq, Time] → [B, 1, Freq, Time]
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 献上指数衰减的学习率
        scheduler.step()
        for param_group in optimizer.param_groups:
            if param_group['lr'] < MIN_LEARNING_RATE:
                param_group['lr'] = MIN_LEARNING_RATE
        
        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total * 100
        logger.info(f"轮次 [{epoch+1}/{NUM_EPOCHS}] 损失: {epoch_loss:.4f} 训练集准确率: {epoch_acc:.2f}% 本轮学习率: {optimizer.param_groups[0]['lr']:.6f}")

    model_path = MODEL_PATH
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存")