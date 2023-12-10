import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch.nn.functional as F
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

# 加载模型
model_path = "checkpoints/best.pth"
model = CustomCNN(num_classes=26)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# 字母列表
letters = [chr(ord('a') + i) for i in range(26)]

# 图像预处理函数
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# 进行字母预测
def predict_letter(image, model):
    image = transform(image).unsqueeze(0)
    output = model(image)
    probabilities = F.softmax(output, dim=1)
    _, predicted_idx = torch.max(output, dim=1)
    predicted_letter = letters[predicted_idx]
    confidence = torch.max(probabilities).item()
    return predicted_letter, confidence


# Web应用
st.title("手写字母识别应用")

# 创建可绘制的画布
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("识别"):
    if canvas_result.image_data is not None:
        # 将绘制的图像转换为PIL Image
        image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        # 进行字母识别
        predicted_letter, confidence = predict_letter(image, model)
        st.write(f"预测结果: {predicted_letter}")
        st.write(f"置信度: {confidence}")
