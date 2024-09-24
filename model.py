import torch
import param
import torch.nn as nn
from transformers import BertTokenizer, BertModel, DistilBertModel, RobertaModel, AutoImageProcessor, \
    ResNetForImageClassification, ViTImageProcessor, ViTModel
from torchvision import datasets, transforms
from torchvision import models


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained('./models/bert-base-uncased')

    def forward(self, texts_inputs, attention_mask):
        text_outputs = self.encoder(input_ids=texts_inputs, attention_mask=attention_mask)
        cls_embedding_t = text_outputs.last_hidden_state[:, 0, :]
        return cls_embedding_t


class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.model = ViTModel.from_pretrained('models/vit-base-patch16-224')

    def forward(self, img_inputs):
        img_outputs = self.model(**img_inputs)
        # 提取 [CLS] token 的特征表示
        cls_embedding_i = img_outputs.last_hidden_state[:, 0, :]
        return cls_embedding_i


class DistilBertEncoder(nn.Module):
    def __init__(self):
        super(DistilBertEncoder, self).__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        feat = self.pooler(pooled_output)
        return feat


class RobertaEncoder(nn.Module):
    def __init__(self):
        super(RobertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        feat = sequence_output[:, 0, :]
        return feat


class DistilRobertaEncoder(nn.Module):
    def __init__(self):
        super(DistilRobertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('distilroberta-base')
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        feat = sequence_output[:, 0, :]
        return feat


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 改为3分类
        param.num_labels = 3
        self.classifier = nn.Linear(param.hidden_size, param.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, x):
        x = self.dropout(x)
        out = self.classifier(x)
        return out

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaClassifier(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, dropout=0.1):
        super(RobertaClassifier, self).__init__()
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(param.hidden_size, param.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.pooler(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(param.hidden_size, param.intermediate_size),
            nn.LeakyReLU(),
            nn.Linear(param.intermediate_size, param.intermediate_size),
            nn.LeakyReLU(),
            nn.Linear(param.intermediate_size, 1),
            #nn.Sigmoid()
        )

    def forward(self, x):
        """Forward the discriminator."""
        out = self.layer(x)
        return out
