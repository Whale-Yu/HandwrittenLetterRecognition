import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 定义自定义CNN网络
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def predict(model_path, image_path):
    # 加载保存的模型
    model = CustomCNN(num_classes=26)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 图像预处理函数
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载输入图像并进行预处理
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # 使用模型进行预测
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_idx = torch.max(output, dim=1)
        predicted_label = chr(ord('a') + predicted_idx.item())

    # 打印预测结果
    print("预测结果:", predicted_label)


if __name__ == '__main__':
    # 训练好的模型路径
    model_path = "checkpoints/best.pth"

    # 待预测图像路径
    image_path = "testImage/aa.png"

    predict(model_path, image_path)
