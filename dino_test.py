import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn

# 设置设备为GPU，如果不可用则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# 加载预训练的图像处理器
processor = AutoImageProcessor.from_pretrained('./dinov2-base',
                                                local_files_only=True)

# 加载预训练的模型，并将其移至相应的设备（GPU或CPU）
model = AutoModel.from_pretrained('./dinov2-base',
                                  local_files_only=True).to(device)

# 打开第一张图片
image1 = Image.open('sat.png')

# 不计算梯度，用于推断
with torch.no_grad():
    # 使用处理器处理图像，转换为PyTorch张量，并移至相应设备
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    # 通过模型获取输出
    outputs1 = model(**inputs1)
    # 获取最后一层的隐藏状态
    image_features1 = outputs1.last_hidden_state
    # 对特征取平均，以得到单个向量表示
    image_features1 = image_features1.mean(dim=1)
    print(f"outputs1 is {image_features1.shape}")

# 重复上述过程，处理第二张图片
image2 = Image.open('sat.png')
with torch.no_grad():
    inputs2 = processor(images=image2, return_tensors="pt").to(device)
    outputs2 = model(**inputs2)
    image_features2 = outputs2.last_hidden_state
    image_features2 = image_features2.mean(dim=1)

# 使用余弦相似度计算两个向量的相似度
cos = nn.CosineSimilarity(dim=0)
sim = cos(image_features1[0],image_features2[0]).item()
# 将相似度值调整到[0,1]范围内
sim = (sim+1)/2
print('Similarity:', sim)