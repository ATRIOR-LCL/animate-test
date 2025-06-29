import os
import time
from flask import Flask, request, render_template, send_from_directory
from configs.imagenet import imagenent_dict
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()

with open("imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 预测函数
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

        _, predicted = torch.max(outputs, 1)

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
