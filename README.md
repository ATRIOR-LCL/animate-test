# 智能动物识别系统

## 一、快速开始

### 1. 项目依赖
- Python3.12 
- torch
- torchvision
- pillow
- flask

### 2. 启动项目

创建虚拟环境并安装所有依赖
```bash
$ cd animate-test

$ virtualenv .venv

$ source .venv/bin/activate

$ pip install torch torchvision pillow flask
```

启动项目
```bash
$ python app.py
```

## 二、图像识别

### 1. **预训练模型加载（ResNet-18）**

```python
model = models.resnet18(pretrained=True).to(device).eval()
```

* **`resnet18`** 是一个经典的卷积神经网络模型（CNN），来自 ResNet 系列。
* **`pretrained=True`** 表示使用已经在 ImageNet 上训练好的参数（weights）。
* **`.to(device)`** 将模型加载到 CPU 或 GPU 上。
* **`.eval()`** 将模型设置为评估模式（禁用 dropout、batchnorm 的训练行为）。

---

### 2. **图像预处理**

```python
transform = transforms.Compose([
    transforms.Resize(256),              # 调整图像大小为256x256
    transforms.CenterCrop(224),          # 裁剪中心的224x224区域（ResNet 输入尺寸）
    transforms.ToTensor(),               # 转换为张量，范围[0,1]
    transforms.Normalize(                # 标准化：让输入符合模型训练时的分布
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

预处理的目的是确保上传图片的格式、大小和数据分布与模型训练时一致。

---

### 3. **图像预测函数**

```python
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")    # 打开图片并转换为 RGB
    img_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度后放到设备上

    with torch.no_grad():                            # 不进行梯度计算（节省内存与时间）
        outputs = model(img_tensor)                  # 前向传播
        _, predicted = torch.max(outputs, 1)         # 取预测得分最大的类别索引
        label = labels[predicted.item()]             # 获取对应的标签
    return label
```

这个函数是识别的核心，完成了以下步骤：

* 加载并预处理图片；
* 将图片送入模型得到输出；
* 选择得分最高的分类；
* 返回该类别的标签（如 `golden retriever`、`zebra` 等）。

---

## 三、人工智能算法

### 1. **卷积神经网络（CNN）**

模型用的是 ResNet-18，本质是一个 **深层卷积神经网络**，其结构包括：

* 卷积层（提取特征）；
* 残差结构（ResNet 的创新）；
* 全连接层（用于分类）。

#### ResNet 关键特点：

* 引入 **残差连接**（shortcut connections），解决深层网络中的梯度消失问题。
* 比如：`y = F(x) + x`，直接跳过若干层，把输入加到输出上。

### 2. **ImageNet 数据集**

* ResNet-18 是在 **ImageNet** 上训练的，这是一个大规模图像数据集，包含 **1000 个分类** 和 **超过百万张图片**。
* 所以模型可以识别常见的动物、工具、植物等物品。

### 3. **Softmax + Cross Entropy Loss（训练阶段）**

虽然代码没有涉及训练，但你要知道模型在训练时：

* 输出是一个长度为 1000 的向量，每个值代表一个类别的预测概率（通过 softmax）；
* 使用交叉熵损失函数来优化参数，使模型输出更准确。

---

## 三、预测流程总结图

```
上传图像
    ↓
读取并转换为RGB → resize → crop → tensor → normalize
    ↓
送入ResNet-18（1000分类的 CNN 模型）
    ↓
输出概率向量（如：[0.01, 0.02, ..., 0.8, ...]）
    ↓
选取概率最大的类别索引
    ↓
通过 labels[index] 转换为标签（如：'golden retriever'）
    ↓
返回标签，渲染 HTML 页面显示结果
```
