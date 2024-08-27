"""Adversarial adaptation to train target encoder."""

import torch
from utils import make_cuda
import torch.nn.functional as F
import torch.nn as nn
import param
import torch.optim as optim
from utils import save_model, save_model_encoder
from modules.attention import CrsAtten
from transformers import ViTImageProcessor, ViTModel, RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
from modules.my_resnet import MyResnet_wofc
from modules.ImgTextMatching import ImgTextMatching
from cross_attention import CrossAttention  # 导入 CrossAttention 类

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Running on GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Running on CPU.")


def pretrain(args, encoder, classifier, data_loader):
    """Train classifier for source domain."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    for epoch in range(args.pre_epochs):
        for step, (reviews, mask, labels) in enumerate(data_loader):
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            labels = make_cuda(labels)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for discriminator
            feat = encoder(reviews, mask)
            preds = classifier(feat)
            cls_loss = CELoss(preds, labels)

            # optimize source classifier
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))

    # save final model
    save_model(args, encoder, param.src_encoder_path)
    save_model(args, classifier, param.src_classifier_path)

    return encoder, classifier


def multi_pretrain(args, encoder_t, encoder_i, classifier, data_loader):
    """Train classifier for source domain."""
    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder_t.parameters()) + list(encoder_i.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder_t.train()
    encoder_i.train()
    classifier.train()

    for epoch in range(args.pre_epochs):
        for step, (reviews, masks, img_datas, labels) in enumerate(data_loader):

            text_feats = make_cuda(reviews)
            masks = make_cuda(masks)
            img_feats = make_cuda(img_datas)

            img_feats = {'pixel_values': img_feats}
            labels = make_cuda(labels)
            # zero gradients for optimizer
            optimizer.zero_grad()

            text_outputs = encoder_t(text_feats, masks)
            image_outputs = encoder_i(img_feats)

            # 初始化交叉注意力层
            cross_attention = CrossAttention(hidden_size=768).to(device)

            # 将文本表示作为 key 和 value，图像表示作为 query 进行交叉注意力融合
            fused_representation_i2t = cross_attention(query=image_outputs.unsqueeze(1).to(device),
                                                       key=text_outputs.unsqueeze(1).to(device),
                                                       value=text_outputs.unsqueeze(1).to(device)).to(device)

            fused_representation_t2i = cross_attention(query=text_outputs.unsqueeze(1).to(device),
                                                       key=image_outputs.unsqueeze(1).to(device),
                                                       value=image_outputs.unsqueeze(1).to(device)).to(device)

            # 融合两个交叉注意力的结果
            # 这里采用相加，拼接或线性层(线性层融合)
            # 拼接两个表示，得到形状为 [batch_size, 1536]
            fused_input = torch.cat((fused_representation_i2t, fused_representation_t2i), dim=-1).to(device)

            # 定义一个线性层，将 1536 维的输入映射到 128 维的输出
            fusion_layer = torch.nn.Linear(1536, 128).to(device)

            # 通过线性层获得融合后的特征
            fused_representation = fusion_layer(fused_input)
            fused_representation = fused_representation.squeeze(1)

            preds = classifier(fused_representation)

            cls_loss = CELoss(preds, labels)

            # optimize source classifier
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))

    # save final model
    save_model_encoder(args, encoder_t, encoder_i, param.src_encoder_path_t, param.src_encoder_path_i)
    save_model(args, classifier, param.src_classifier_path)

    return encoder_t, encoder_i, classifier


def adapt(args, src_encoder, tgt_encoder, discriminator,
          src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_all_loader):
    """Train encoder for target domain."""

    # set train state for Dropout and BN layers
    src_encoder.eval()
    src_classifier.eval()
    tgt_encoder.train()
    discriminator.train()

    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    # BCELoss = nn.CrossEntropyLoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    optimizer_G = optim.Adam(tgt_encoder.parameters(), lr=param.d_learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=param.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
        for step, ((reviews_src, src_mask, _), (reviews_tgt, tgt_mask, _)) in data_zip:
            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)

            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)

            # zero gradients for optimizer
            optimizer_D.zero_grad()

            # extract and concat features
            with torch.no_grad():
                feat_src = src_encoder(reviews_src, src_mask)
            feat_src_tgt = tgt_encoder(reviews_src, src_mask)
            feat_tgt = tgt_encoder(reviews_tgt, tgt_mask)
            feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)

            # predict on discriminator
            pred_concat = discriminator(feat_concat.detach())

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src_tgt.size(0))).unsqueeze(1)
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0))).unsqueeze(1)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for discriminator
            dis_loss = BCELoss(pred_concat, label_concat)
            dis_loss.backward()

            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            # optimize discriminator
            optimizer_D.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_G.zero_grad()
            T = args.temperature

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = F.softmax(src_classifier(feat_src) / T, dim=-1)
            tgt_prob = F.log_softmax(src_classifier(feat_src_tgt) / T, dim=-1)
            kd_loss = KLDivLoss(tgt_prob, src_prob.detach()) * T * T

            # 调整label_src大小，使其与pred_tgt一致
            # if pred_tgt.size(0) != label_src.size(0):
            #     label_src = make_cuda(torch.zeros(pred_tgt.size(0), 1).float())

            # compute loss for target encoder
            gen_loss = BCELoss(pred_tgt, label_src)
            loss_tgt = args.alpha * gen_loss + args.beta * kd_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), args.max_grad_norm)
            # optimize target encoder
            optimizer_G.step()

            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "acc=%.4f g_loss=%.4f d_loss=%.4f kd_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         acc.item(),
                         gen_loss.item(),
                         dis_loss.item(),
                         kd_loss.item()))

        evaluate(tgt_encoder, src_classifier, tgt_data_all_loader)

    return tgt_encoder


def multi_adapt(args, src_encoder_t, src_encoder_i, tgt_encoder_t, tgt_encoder_i, discriminator,
                src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_all_loader):
    """Train encoder for target domain."""

    # set train state for Dropout and BN layers
    src_encoder_t.eval()
    src_encoder_i.eval()
    src_classifier.eval()

    tgt_encoder_t.train()
    tgt_encoder_i.train()
    discriminator.train()

    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    # BCELoss = nn.CrossEntropyLoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    optimizer_G = optim.Adam(list(tgt_encoder_t.parameters()) + list(tgt_encoder_i.parameters()),
                             lr=param.d_learning_rate)
    # optimizer_Gt = optim.Adam(tgt_encoder_t.parameters(), lr=param.d_learning_rate)
    # optimizer_Gi = optim.Adam(tgt_encoder_i.parameters(), lr=param.d_learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=param.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
        for step, ((reviews_src, src_mask, src_img, _), (reviews_tgt, tgt_mask, tgt_img, _)) in data_zip:
            # 检查 reviews_src 和 reviews_tgt 的第一个维度是否一致
            if reviews_src.size(0) != reviews_tgt.size(0):
                continue  # 如果不一致，则跳过当前批次

            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)
            src_img = make_cuda(src_img)
            src_img = {'pixel_values': src_img}

            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)
            tgt_img = make_cuda(tgt_img)
            tgt_img = {'pixel_values': tgt_img}

            # zero gradients for optimizer
            optimizer_D.zero_grad()

            # extract and concat features
            with torch.no_grad():
                # ********** 源域 ************#
                feat_text_src = src_encoder_t(reviews_src, src_mask)
                feat_img_src = src_encoder_i(src_img)

                # 初始化交叉注意力层
                cross_attention = CrossAttention(hidden_size=768).to(device)

                # 将文本表示作为 key 和 value，图像表示作为 query 进行交叉注意力融合
                fused_representation_i2t = cross_attention(query=feat_img_src.unsqueeze(1).to(device),
                                                           key=feat_text_src.unsqueeze(1).to(device),
                                                           value=feat_text_src.unsqueeze(1).to(device)).to(device)

                fused_representation_t2i = cross_attention(query=feat_text_src.unsqueeze(1).to(device),
                                                           key=feat_img_src.unsqueeze(1).to(device),
                                                           value=feat_img_src.unsqueeze(1).to(device)).to(device)

                # 融合两个交叉注意力的结果
                # 这里采用相加，拼接或线性层(线性层融合)
                # 拼接两个表示，得到形状为 [batch_size, 1536]
                fused_input = torch.cat((fused_representation_i2t, fused_representation_t2i), dim=-1).to(device)

                # 定义一个线性层，将 1536 维的输入映射到 128 维的输出
                fusion_layer = torch.nn.Linear(1536, 128).to(device)

                # 通过线性层获得融合后的特征
                fused_representation = fusion_layer(fused_input)
                feat_src = fused_representation.squeeze(1)

                # *******************************#
            # ********** 源域、目标域 ************#
            feat_text_src_tgt = tgt_encoder_t(reviews_src, src_mask)
            feat_img_src_tgt = tgt_encoder_i(src_img)

            # 初始化交叉注意力层
            cross_attention = CrossAttention(hidden_size=768).to(device)

            # 将文本表示作为 key 和 value，图像表示作为 query 进行交叉注意力融合
            fused_representation_i2t = cross_attention(query=feat_img_src_tgt.unsqueeze(1).to(device),
                                                       key=feat_text_src_tgt.unsqueeze(1).to(device),
                                                       value=feat_text_src_tgt.unsqueeze(1).to(device)).to(device)

            fused_representation_t2i = cross_attention(query=feat_text_src_tgt.unsqueeze(1).to(device),
                                                       key=feat_img_src_tgt.unsqueeze(1).to(device),
                                                       value=feat_img_src_tgt.unsqueeze(1).to(device)).to(device)

            # 融合两个交叉注意力的结果
            # 这里采用相加，拼接或线性层(线性层融合)
            # 拼接两个表示，得到形状为 [batch_size, 1536]
            fused_input = torch.cat((fused_representation_i2t, fused_representation_t2i), dim=-1).to(device)

            # 定义一个线性层，将 1536 维的输入映射到 128 维的输出
            fusion_layer = torch.nn.Linear(1536, 128).to(device)

            # 通过线性层获得融合后的特征
            fused_representation = fusion_layer(fused_input)
            feat_src_tgt = fused_representation.squeeze(1)

            # *******************************#
            # ********** 目标域 ************#
            feat_text_tgt = tgt_encoder_t(reviews_tgt, tgt_mask)
            feat_img_tgt = tgt_encoder_i(tgt_img)

            # 初始化交叉注意力层
            cross_attention = CrossAttention(hidden_size=768).to(device)

            # 将文本表示作为 key 和 value，图像表示作为 query 进行交叉注意力融合
            fused_representation_i2t = cross_attention(query=feat_img_src_tgt.unsqueeze(1).to(device),
                                                       key=feat_text_src_tgt.unsqueeze(1).to(device),
                                                       value=feat_text_src_tgt.unsqueeze(1).to(device)).to(device)

            fused_representation_t2i = cross_attention(query=feat_text_src_tgt.unsqueeze(1).to(device),
                                                       key=feat_img_src_tgt.unsqueeze(1).to(device),
                                                       value=feat_img_src_tgt.unsqueeze(1).to(device)).to(device)

            # 融合两个交叉注意力的结果
            # 这里采用相加，拼接或线性层(线性层融合)
            # 拼接两个表示，得到形状为 [batch_size, 1536]
            fused_input = torch.cat((fused_representation_i2t, fused_representation_t2i), dim=-1).to(device)

            # 定义一个线性层，将 1536 维的输入映射到 128 维的输出
            fusion_layer = torch.nn.Linear(1536, 128).to(device)

            # 通过线性层获得融合后的特征
            fused_representation = fusion_layer(fused_input)
            feat_tgt = fused_representation.squeeze(1)

            # *******************************#
            feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)

            # predict on discriminator
            pred_concat = discriminator(feat_concat.detach())

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src_tgt.size(0))).unsqueeze(1)
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0))).unsqueeze(1)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for discriminator
            dis_loss = BCELoss(pred_concat, label_concat)
            dis_loss.backward()

            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            # optimize discriminator
            optimizer_D.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_G.zero_grad()
            T = args.temperature

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = F.softmax(src_classifier(feat_src) / T, dim=-1)
            tgt_prob = F.log_softmax(src_classifier(feat_src_tgt) / T, dim=-1)
            kd_loss = KLDivLoss(tgt_prob, src_prob.detach()) * T * T

            # 调整label_src大小，使其与pred_tgt一致
            # if pred_tgt.size(0) != label_src.size(0):
            #     label_src = make_cuda(torch.zeros(pred_tgt.size(0), 1).float())

            # compute loss for target encoder
            gen_loss = BCELoss(pred_tgt, label_src)
            loss_tgt = args.alpha * gen_loss + args.beta * kd_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(tgt_encoder_t.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(tgt_encoder_i.parameters(), args.max_grad_norm)

            # optimize target encoder
            optimizer_G.step()

            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "acc=%.4f g_loss=%.4f d_loss=%.4f kd_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         acc.item(),
                         gen_loss.item(),
                         dis_loss.item(),
                         kd_loss.item()))

        multi_evaluate(tgt_encoder_t, tgt_encoder_i, src_classifier, tgt_data_all_loader)

    return tgt_encoder_t, tgt_encoder_i


def evaluate(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (reviews, mask, labels) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        labels = make_cuda(labels)

        with torch.no_grad():
            feat = encoder(reviews, mask)
            preds = classifier(feat)
        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))

    return acc


def multi_evaluate(encoder_t, encoder_i, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder_t.eval()
    encoder_i.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (reviews, masks, img_datas, labels) in data_loader:
        labels = make_cuda(labels)

        with torch.no_grad():
            text_feats = make_cuda(reviews)
            masks = make_cuda(masks)
            img_feats = make_cuda(img_datas)

            img_feats = {'pixel_values': img_feats}

            text_outputs = encoder_t(text_feats, masks)
            image_outputs = encoder_i(img_feats)

            # 初始化交叉注意力层
            cross_attention = CrossAttention(hidden_size=768).to(device)

            # 将文本表示作为 key 和 value，图像表示作为 query 进行交叉注意力融合
            fused_representation_i2t = cross_attention(query=image_outputs.unsqueeze(1).to(device),
                                                       key=text_outputs.unsqueeze(1).to(device),
                                                       value=text_outputs.unsqueeze(1).to(device)).to(device)

            fused_representation_t2i = cross_attention(query=text_outputs.unsqueeze(1).to(device),
                                                       key=image_outputs.unsqueeze(1).to(device),
                                                       value=image_outputs.unsqueeze(1).to(device)).to(device)

            # 融合两个交叉注意力的结果
            # 这里采用相加，拼接或线性层(线性层融合)
            # 拼接两个表示，得到形状为 [batch_size, 1536]
            fused_input = torch.cat((fused_representation_i2t, fused_representation_t2i), dim=-1).to(device)

            # 定义一个线性层，将 1536 维的输入映射到 128 维的输出
            fusion_layer = torch.nn.Linear(1536, 128).to(device)

            # 通过线性层获得融合后的特征
            fused_representation = fusion_layer(fused_input)
            fused_representation = fused_representation.squeeze(1)

            preds = classifier(fused_representation)

        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))

    return acc
