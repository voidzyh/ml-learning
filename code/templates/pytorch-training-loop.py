"""
PyTorch 标准训练循环模板
用法: 根据你的任务修改 Model / Dataset / 超参数
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# import wandb  # 取消注释以启用W&B记录

# ─── 超参数 ───
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── 模型定义 ───
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: 定义你的网络层
        pass

    def forward(self, x):
        # TODO: 定义前向传播
        pass


# ─── 训练一个epoch ───
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        # 分类任务: 计算准确率
        _, predicted = output.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─── 评估 ───
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        output = model(batch_x)
        loss = criterion(output, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─── 主训练循环 ───
def main():
    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # TODO: 创建你的DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        # scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✅ 新最佳! Val Acc: {val_acc:.4f}")

    print(f"\n🏆 最佳验证准确率: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
