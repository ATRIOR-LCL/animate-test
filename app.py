import os
import time
from flask import Flask, request, render_template, send_from_directory
from configs.imagenet import imagenent_dict

# 用于打开和处理上传的照片
from PIL import Image

# PyTorch 主模块，用于张量操作、模型加载与推理
import torch

# 对图像进行缩放、裁剪、归一化等转换
import torchvision.transforms as transforms

# 包含与训练的视觉模型 ResNet-18
import torchvision.models as models


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 静态文件路由
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型与标签
# 判断是否可用 GPU（cuda），如果不可用则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 ResNet-18 模型，权重来自于 ImageNet 数据集
# to(device) 将模型移动到选定设备（CPU、GPU）
# eval() 表示设置为推理模式，禁用 dropout、batchnorm 等训练行为
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()

# 从 imagenet_classes.txt 中读取每一行，作为模型输出的分类标签
with open("imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 图像预处理
transform = transforms.Compose([
    # 将短边缩放到 256 像素
    transforms.Resize(256),
    # 从中心裁剪 224x224 像素，符合 ResNet 的输入尺寸
    transforms.CenterCrop(224),
    # 将图像转换为张量，值范围为 [0,1]
    transforms.ToTensor(),
    # 对张量进行归一化，减去均值并除以标准差（ImageNet 数据集上的统计值）
    # 输出像素= (输入像素 - mean) / std
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 每个通道的均值
        std=[0.229, 0.224, 0.225]   # 每个通道的标准差
    )
])

# 预测函数
def predict_image(image_path):
    # 打开图片，转换为 RGB，防止灰度图或带 alpha 通道的图片错误
    # ResNet 要求的张量维度是 (3, H, W) 而灰度图的张量维度是 (1, H, W) 也就是只有一个通道，对于彩色图片均有 RGB 三个通道，所以灰度图并不适用该项目的图像识别
    # 提前转为 RGB，是在“喂网络吃饭前，先把食物做成它能吃的形状”。
    image = Image.open(image_path).convert("RGB")

    # unsqueeze 增加一个 batch 维度，在第 0 个维度添加一个维度，把 shape 从 (3, 224, 224) 变成 (1, 3, 224, 224)
    # PyTorch、TensorFlow 统一都设计为 (N, C, H, W)，即使 N=1 也要保留这个维度。
    img_tensor = transform(image).unsqueeze(0).to(device) # to(device) 将张量转移到同样的设备

    # 关闭梯度计算，提高推理速度
    # 在“预测”或“评估”阶段，我们只需要得到模型的输出，而不需要进行 反向传播 或 梯度更新，所以可以关闭梯度计算，节省资源。
    with torch.no_grad():
        # model(img_tensor) 将图像张量输入模型，放回为一个长度为 1000 的向量（每个类别的 logits）
        outputs = model(img_tensor)

        # 取出每张图片预测结果中概率最大的一个索引
        _, predicted = torch.max(outputs, 1)

        # 根据索引找到对应的 ImageNet 名称，作为最终预测标签
        label = labels[predicted.item()]
    return label

# 首页路由
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', label='No file part')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', label='No selected file')

        if file:
            file_extension = os.path.splitext(file.filename)[1]
            timestamp = str(int(time.time()))
            new_filename = f"{timestamp}{file_extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(filepath)
            label = predict_image(filepath)
            return render_template('index.html', label=imagenent_dict.get(label), image_url=filepath)

    return render_template('index.html', label=None)

if __name__ == '__main__':
    app.run(debug=True)
