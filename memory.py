import torch
from transformers import BertTokenizer, BertModel

# 设置设备为CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载BERT tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')
model = BertModel.from_pretrained('models/bert-base-uncased').to(device)

# 假设你有一个文本列表
texts = [
    "Replace me by any text you'd like.",
    "This is another example sentence.",
    "And yet another sentence."
]

# 批量编码文本
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)

# 通过模型前向传播获取输出
with torch.no_grad():  # 禁用梯度计算以节省内存
    outputs = model(**encoded_inputs)

# 提取最后一层的[CLS] token特征
cls_features = outputs.last_hidden_state[:, 0, :]

# 输出结果
print(cls_features)