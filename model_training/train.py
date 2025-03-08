import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from PIL import Image, ImageFile
from timm.scheduler import CosineLRScheduler  # 需要安装pip install timm
from sklearn.model_selection import train_test_split  # scikit-learn
import matplotlib.pyplot as plt
import os
import json
import warnings
from datetime import datetime
from check_image import check_images


ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像
# 忽略 PIL 的 EXIF 警告
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

print(f"当前使用的设备：{'GPU' if torch.cuda.is_available() else 'CPU'}")
device_hardware = 'cuda' if torch.cuda.is_available() else 'cpu'


# 硬件优化配置
class Config:
    data_dir = "./垃圾图片库"  # 包含子目录的结构化数据集路径
    num_classes = len(os.listdir(data_dir))  # 自动获取类别数量（根据子文件夹数量）
    batch_size = 8  # 调整为适合8G显存的批次大小
    img_size = 224  # 根据显存情况可降为192
    num_workers = 4  # 根据CPU核心数优化
    lr = 2e-4  # 更保守的学习率
    epochs = 30
    device = torch.device(device_hardware)
    save_path = "./garbage_classifier_best_EfficientNet-B4.pth"
    label_mapping_path = "./static/data/class_indices.json"  # 保存类别映射关系


# 增强的数据预处理（适配细粒度分类）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(Config.img_size, scale=(0.5, 1.0)),  # 扩大裁剪范围
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.RandomRotation(25),
    transforms.RandomAffine(degrees=0, shear=10),  # 新增错切变换
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(int(Config.img_size * 1.1)),
    transforms.CenterCrop(Config.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def custom_loader(path):
    try:
        img = Image.open(path)
        img._getexif = lambda: None  # 禁用 EXIF 数据读取
        img = img.convert("RGB")
    except (IOError, OSError, SyntaxError) as e:
        print(f"警告：图像文件损坏，跳过该文件：{path}")
        # 如果图像损坏，返回一个默认的占位图像
        img = Image.new("RGB", (Config.img_size, Config.img_size), color=(255, 255, 255))
    return img


# 加载数据集并保存类别映射
full_dataset = datasets.ImageFolder(root=Config.data_dir, loader=custom_loader)
class_names = full_dataset.classes
class_to_idx = full_dataset.class_to_idx

# 保存类别映射关系
with open(Config.label_mapping_path, 'w') as f:
    json.dump({'class_to_idx': class_to_idx, 'classes': class_names}, f)

# 分层划分训练验证集（需要sklearn）
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=full_dataset.targets,
    random_state=42
)

# 创建子数据集
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)


# 应用transform的封装类
class ApplyTransform:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)


# 应用不同的transform
train_dataset = ApplyTransform(train_dataset, train_transform)
val_dataset = ApplyTransform(val_dataset, val_transform)

# 优化后的数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=Config.num_workers,
    pin_memory=True  # 加速数据传输
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=Config.num_workers,
    pin_memory=True
)


# 使用更轻量的模型（适合8G显存）
def create_model():
    """
    模型选择:
        弱: EfficientNet-B0  EfficientNet-B1 ==> 这些模型参数量更少，训练速度更快。||  EfficientNet-B3 || MobileNetV3-Small MobileNetV3-Large ==> 专为移动端设计，计算效率高。
        强: EfficientNet-B4 ==> 在更大的数据集上表现更好。|| ResNet-50 ResNet-101 ConvNeXt ==> 经典的卷积神经网络架构，适合细粒度分类任务。 || Swin Transformer ==> 基于Transformer的现代架构，通常比传统CNN表现更好。
    函数:
        创建模型
    :return:创建的模型
    """
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')  # 平衡速度和精度
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),  # 提升Dropout比例
        nn.Linear(in_features, Config.num_classes)
    )
    return model


model = create_model().to(Config.device)

# 混合精度训练和梯度缩放
scaler = torch.amp.GradScaler(device=Config.device)

# 优化器和学习率调度
optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-3)  # 增强权重衰减
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=Config.epochs,
    warmup_t=5,  # 延长预热阶段
    warmup_lr_init=1e-6,
)


# 早停机制
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


early_stopper = EarlyStopping(patience=10, delta=0.005)


# 训练循环
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0

    for inputs, labels in loader:
        inputs = inputs.to(Config.device, non_blocking=True)
        labels = labels.to(Config.device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_hardware):
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

    return total_loss / len(loader.dataset), correct.double() / len(loader.dataset)


def validate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(Config.device, non_blocking=True)
            labels = labels.to(Config.device, non_blocking=True)

            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

    return total_loss / len(loader.dataset), correct.double() / len(loader.dataset)


def plotting_loss_curve(train_losses, val_losses):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(f'./training_curve_{timestamp}.png')


# 主训练函数
def main():
    best_acc = 0.0
    train_losses = []
    val_losses = []

    for epoch in range(Config.epochs):
        print(f'Epoch {epoch + 1}/{Config.epochs}')

        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, optimizer)
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader)
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}')

        # 学习率调度
        scheduler.step(epoch + 1)

        # 早停检查
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, Config.save_path)
            print(f'Saved new best model with acc: {best_acc:.4f}')

        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # 绘制损失曲线（可选）
        if epoch == Config.epochs - 1 or early_stopper.early_stop:
            plotting_loss_curve(train_losses, val_losses)

    print(f'Training complete. Best val Acc: {best_acc:.4f}')


if __name__ == '__main__':
    # Windows 多进程保护
    torch.multiprocessing.freeze_support()
    check_images(Config.data_dir)
    main()
