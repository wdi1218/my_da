from transformers import ViTImageProcessor, ViTModel , RobertaTokenizer, RobertaModel
import torchvision.transforms as transforms
import torch
from PIL import Image
import requests
from cross_attention import CrossAttention  # 导入 CrossAttention 类

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Running on GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Running on CPU.")

# 获取并加载图像
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# 获取并加载本地图像
image_path = 'datasets/IJCAI2019_data/twitter2015_images/975807.jpg'  # 替换为本地图像的路径
image = Image.open(image_path)
text = "Replace me by any text you'd like."

# 确保图像被正确加载
if image.mode != "RGB":
    image = image.convert("RGB")

# 加载处理器和预训练模型
processor = ViTImageProcessor.from_pretrained('models/vit-base-patch16-224')
img_model = ViTModel.from_pretrained('models/vit-base-patch16-224').to(device)  # 注意这里改为 ViTModel 而不是 ViTForImageClassification

# 加载 RoBERTa 分词器和预训练模型
tokenizer = RobertaTokenizer.from_pretrained('models/roberta-base')
text_model = RobertaModel.from_pretrained('models/roberta-base').to(device)


img_inputs = processor(images=image, return_tensors="pt").to(device)
text_input = tokenizer(text, return_tensors='pt').to(device)
print(text_input)
# 前向传播，获取模型输出
img_outputs = img_model(**img_inputs)

text_outputs = text_model(**text_input)


# 提取[CLS] token的特征表示作为图像的全局特征
cls_embedding_i = img_outputs.last_hidden_state[:, 0, :].to(device)   # [batch_size, hidden_size]
cls_embedding_t = text_outputs.last_hidden_state[:, 0, :].to(device)   # [batch_size, hidden_size]

print("Image CLS embedding shape:", cls_embedding_i.shape)
# print("CLS embedding:", cls_embedding_i)

print("Text CLS embedding shape:", cls_embedding_t.shape)
# print("CLS embedding:", cls_embedding_t)


# 初始化交叉注意力层
cross_attention = CrossAttention(hidden_size=768).to(device)

# 将文本表示作为 key 和 value，图像表示作为 query 进行交叉注意力融合
fused_representation_i2t = cross_attention(query=cls_embedding_i.unsqueeze(1).to(device),
                                       key=cls_embedding_t.unsqueeze(1).to(device),
                                       value=cls_embedding_t.unsqueeze(1).to(device)).to(device)

fused_representation_t2i = cross_attention(query=cls_embedding_t.unsqueeze(1).to(device),
                                       key=cls_embedding_i.unsqueeze(1).to(device),
                                       value=cls_embedding_i.unsqueeze(1).to(device)).to(device)

# 融合两个交叉注意力的结果
# 这里采用相加，拼接或线性层(线性层融合)
# 拼接两个表示，得到形状为 [batch_size, 1536]
fused_input = torch.cat((fused_representation_i2t, fused_representation_t2i), dim=-1).to(device)

# 定义一个线性层，将 1536 维的输入映射到 768 维的输出
fusion_layer = torch.nn.Linear(1536, 128).to(device)

# 通过线性层获得融合后的特征
fused_representation = fusion_layer(fused_input)
fused_representation = fused_representation.squeeze(1)  # 结果形状将是 [1, 768]

print(fused_representation.shape)