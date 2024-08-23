import torch
import torch.nn as nn
from transformers import BertModel
from modules.attention import CrsAtten
from modules.my_resnet import MyResnet_wofc
from torch.nn.modules.transformer import MultiheadAttention


class ImgTextMatching(nn.Module):
    def __init__(self):
        super(ImgTextMatching, self).__init__()
        # print('Creating CNN instance.')
        self.resnet = MyResnet_wofc()
        self.txt_linear = nn.Linear(768, 128)
        self.img_linear = nn.Linear(2048, 128)
        self.txt_drop = nn.Dropout(p=0.1)
        self.img_drop = nn.Dropout(p=0.1)

        # #1. img_to_txt bs*d      txt_to_img bs*d
        self.img_to_txt_att = CrsAtten()
        self.txt_to_img_att = CrsAtten()

        # 2. transformer_encoder
        self.img_to_txt_multi_att = MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.txt_to_img_multi_att = MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.class_linear = nn.Linear(128, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, text_feats, img_datas):
        pass
