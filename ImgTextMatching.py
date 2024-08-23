import torch
import torch.nn as nn
from transformers import BertModel
from modules.attention import CrsAtten
from modules.my_resnet import MyResnet_wofc
from torch.nn.modules.transformer import MultiheadAttention


class ImgTextMatching(nn.Module):
    def __init__(self, args):
        super(ImgTextMatching, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained('../bert_base_uncased/')
        print('Creating CNN instance.')
        self.resnet = MyResnet_wofc()
        self.txt_linear = nn.Linear(768, args.hidden_dim)
        self.img_linear = nn.Linear(2048, args.hidden_dim)
        self.txt_drop = nn.Dropout(p = 0.1)
        self.img_drop = nn.Dropout(p = 0.1)

        # #1. img_to_txt bs*d      txt_to_img bs*d
        self.img_to_txt_att = CrsAtten(args)
        self.txt_to_img_att = CrsAtten(args)

        #2. transformer_encoder
        self.img_to_txt_multi_att = MultiheadAttention(embed_dim=args.encoder_dim, num_heads=args.multiatt_head_num, batch_first = True)
        self.txt_to_img_multi_att = MultiheadAttention(embed_dim=args.encoder_dim, num_heads=args.multiatt_head_num, batch_first = True)

        self.class_linear = nn.Linear(args.hidden_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img_datas,
                      input_idx,
                      token_type_ids = None,
                      att_mask = None,
                      label = None):
        img_feats = self.resnet.resnet_forward(img_datas)
        bert_out = self.bert(input_ids = input_idx,
                              token_type_ids = token_type_ids,
                              attention_mask = att_mask)
        txt_feats = bert_out.last_hidden_state

        img_feats = self.img_linear(img_feats.view(img_feats.size(0), -1, img_feats.size(3)))
        img_feats = self.img_drop(img_feats)    #bs*49*d

        txt_feats = self.txt_linear(txt_feats)
        txt_feats = self.txt_drop(txt_feats)    #bs*len*d

        # #1. img_to_txt bs*d      txt_to_img bs*d
        #
        # pool_img_feats = torch.mean(img_feats, dim=1)
        # i2t_out = self.img_to_txt_att(txt_feats, pool_img_feats)
        #
        # pool_txt_feats = torch.mean(txt_feats, dim=1)
        # t2i_out = self.txt_to_img_att(img_feats, pool_txt_feats)
        #
        # out = i2t_out +t2i_out

        #2. transformer_encoder

        i2t_out = self.img_to_txt_multi_att(query = txt_feats, key = img_feats, value = img_feats)
        t2i_out = self.txt_to_img_multi_att(query = img_feats, key = txt_feats, value = txt_feats, key_padding_mask = 1- att_mask)
        i2t_out = i2t_out[0]
        t2i_out = t2i_out[0]

        pool_i2t_out = torch.mean(i2t_out, dim=1)
        pool_t2i_out = torch.mean(t2i_out, dim=1)
        out = pool_i2t_out + pool_t2i_out

        out = self.class_linear(out)

        if label is not None:
            loss = self.loss_fn(out, label)
            return loss, out
        else:
            return out