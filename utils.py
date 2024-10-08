import os
import random
import torch.quantization
from transformers import ViTImageProcessor, ViTModel, RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import numpy as np
import pandas as pd
import torch
from lxml import etree
import xml.etree.ElementTree as ET
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import param
import re
from PIL import Image, ImageFile
from torchvision import models, transforms
from cross_attention import CrossAttention  # 导入 CrossAttention 类
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma`")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta`")

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二张GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Running on GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Running on CPU.")

_transforms = []
_transforms.append(transforms.Resize(256))
_transforms.append(transforms.CenterCrop(224))
_transforms.append(transforms.ToTensor())
_transforms.append(
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]))
default_transforms = transforms.Compose(_transforms)


# 标签转换函数
def convert_label(label):
    if label == 'positive':
        return 0
    elif label == 'neutral':
        return 1
    elif label == 'negative':
        return 2


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, text_feat, img_feat, mask_id, label_id):
        self.text_feat = text_feat
        self.img_feat = img_feat
        self.mask_id = mask_id
        self.label_id = label_id


def TWI_CSV2Array(path1, path2, path3):
    try:
        dev_data = pd.read_csv(path1, sep='\t', encoding='utf-8')
        # 提取第二列（index 1）、第三列（index 2）和 第四列（index 3）
        labels1 = dev_data.iloc[:, 1].values.tolist()  # 第二列
        images1 = dev_data.iloc[:, 2].values.tolist()  # 第三列
        strings1 = dev_data.iloc[:, 3].values.tolist()  # 第四列
    except pd.errors.ParserError as e:
        print(print(f"Error parsing {path1}: {e}"))
        labels1, images1, strings1 = [], [], []

    try:
        test_data = pd.read_csv(path2, sep='\t', encoding='utf-8')
        # 提取第二列（index 1）、第三列（index 2）和 第四列（index 3）
        labels2 = test_data.iloc[:, 0].values.tolist()  # 第二列
        images2 = test_data.iloc[:, 1].values.tolist()  # 第三列
        strings2 = test_data.iloc[:, 2].values.tolist()  # 第四列
    except pd.errors.ParserError as e:
        print(print(f"Error parsing {path1}: {e}"))
        labels2, images2, strings2 = [], [], []

    try:
        train_data = pd.read_csv(path3, sep='\t', encoding='utf-8')
        # 提取第二列（index 1）、第三列（index 2）和 第四列（index 3）
        labels3 = train_data.iloc[:, 1].values.tolist()  # 第二列
        images3 = train_data.iloc[:, 2].values.tolist()  # 第三列
        strings3 = train_data.iloc[:, 3].values.tolist()  # 第四列
    except pd.errors.ParserError as e:
        print(print(f"Error parsing {path1}: {e}"))
        labels3, images3, strings3 = [], [], []

    label = np.concatenate((labels1, labels2, labels3))
    image = np.concatenate((images1, images2, images3))
    strings = np.concatenate((strings1, strings2, strings3))

    return label, image, strings


def mvsa_data(path):
    # 初始化列表来存储第二列和第三列的数据
    ids = []
    texts = []
    labels = []

    image_paths = []  # 新增列表来存储图片路径
    # 打开文件并逐行读取
    with open(path, 'r', encoding='utf-8') as file:
        # 跳过第一行（列名）
        # next(file)
        for line in file:
            # print(line)
            # 去除行尾的换行符和空白
            line = line.strip()
            # 分割每行数据
            parts = line.split('\t')
            # 读取第二列和第三列
            id = parts[0]
            text = parts[1]
            label = convert_label(parts[2])

            image_path = f"MVSA/data/{id}.jpg"
            # 添加到列表中
            texts.append(text)
            labels.append(label)
            image_paths.append(image_path)
    return texts, labels, image_paths


def yelp_data(path):
    # 初始化列表来存储第二列和第三列的数据
    ids = []
    texts = []
    labels = []

    image_paths = []  # 新增列表来存储图片路径
    # 打开文件并逐行读取
    with open(path, 'r', encoding='utf-8') as file:
        # 跳过第一行（列名）
        # next(file)
        for line in file:
            # print(line)
            # 去除行尾的换行符和空白
            line = line.strip()
            # 分割每行数据
            parts = line.split('\t')
            # 读取第二列和第三列
            id = parts[1]
            text = parts[2]
            label = convert_label(parts[3])

            image_path = f"shiyan/images/{id}.jpg"
            # 添加到列表中
            texts.append(text)
            labels.append(label)
            image_paths.append(image_path)
    return texts, labels, image_paths


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    else:
        print("cpu")
    return tensor


def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(args, net, restore=None):
    # restore model weights
    if restore is not None:
        path = os.path.join(param.model_root, args.src, args.model, str(args.seed), restore)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
            print("Restore model from: {}".format(os.path.abspath(path)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def init_multi_model(args, net_t, net_i, restore_t=None, restore_i=None):
    # Restore text encoder weights
    if restore_t is not None:
        path_t = os.path.join(param.model_root, args.src, args.model, str(args.seed), restore_t)
        if os.path.exists(path_t):
            net_t.load_state_dict(torch.load(path_t))
            print("Restore text model from: {}".format(os.path.abspath(path_t)))

    # Restore image encoder weights
    if restore_i is not None:
        path_i = os.path.join(param.model_root, args.src, args.model, str(args.seed), restore_i)
        if os.path.exists(path_i):
            net_i.load_state_dict(torch.load(path_i))
            print("Restore image model from: {}".format(os.path.abspath(path_i)))

    # Check if CUDA is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net_t.cuda()
        net_i.cuda()

    return net_t, net_i


def save_model_encoder(args, net_t, net_i, name_t, name_i):
    """Save trained model."""
    folder = os.path.join(param.model_root, args.src, args.model, str(args.seed))
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 保存文本编码器模型
    text_model_path = os.path.join(folder, f"{name_t}_text_model.pth")
    torch.save(net_t.state_dict(), text_model_path)
    print("Saved text encoder model to: {}".format(text_model_path))

    # 保存图像编码器模型
    image_model_path = os.path.join(folder, f"{name_i}_image_model.pth")
    torch.save(net_i.state_dict(), image_model_path)
    print("Saved image encoder model to: {}".format(image_model_path))


def save_model(args, net, name):
    """Save trained model."""
    folder = os.path.join(param.model_root, args.src, args.model, str(args.seed))
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), path)
    print("save pretrained model to: {}".format(path))


def convert_examples_to_features(reviews, labels, max_seq_length, tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]',
                                 pad_token=0):
    features = []
    for ex_index, (review, label) in enumerate(zip(reviews, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(reviews)))
        tokens = tokenizer.tokenize(review)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        tokens = [cls_token] + tokens + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          mask_id=input_mask,
                          label_id=label,
                          img_datas=None))

    return features


def image_data(root_path, image_path, processor, img_model):
    if image_path is None:
        raise ValueError("image_path should not be None")

    full_path = os.path.join(root_path, image_path)
    # 打开并处理图像
    image = Image.open(full_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    img_output = img_model(**img_inputs)
    return img_output


def multi_convert_examples_to_features(reviews, labels, max_seq_length, tokenizer,
                                       cls_token='[CLS]', sep_token='[SEP]', pad_token=0, root_path, image_paths):
    features = []
    tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')
    processor = ViTImageProcessor.from_pretrained('models/vit-base-patch16-224')

    batch_size = 8
    max_text_length = 128  # 确保所有文本的最大长度一致
    img_size = (224, 224)  # ViT 模型输入的图像尺寸

    processed_count = 0  # 初始化计数器

    if isinstance(reviews, np.ndarray):
        reviews = reviews.tolist()

    # 统一处理所有样本的文本
    all_text_inputs = tokenizer(reviews, padding=True, truncation=True, max_length=max_text_length,
                                return_tensors='pt').to(device)
    # 分批处理图像
    for i in range(0, len(image_paths), batch_size):
        batch_image_paths = image_paths[i:i + batch_size]
        batch_images = [Image.open(os.path.join(root_path, img_path)).convert("RGB").resize(img_size) for img_path in
                        batch_image_paths]
        batch_img_inputs = processor(images=batch_images, return_tensors="pt").to(device)

        batch_texts = all_text_inputs['input_ids'][i:i + batch_size]
        batch_masks = all_text_inputs['attention_mask'][i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        for input_id, image_input, mask, label in zip(batch_texts, batch_img_inputs['pixel_values'], batch_masks,
                                                      batch_labels):
            features.append(
                InputFeatures(
                    text_feat=input_id.detach().cpu().numpy(),
                    img_feat=image_input.detach().cpu().numpy(),
                    mask_id=mask.detach().cpu().numpy(),
                    label_id=label
                )
            )

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count * batch_size} 条数据")

    return features


# def multi_convert_examples_to_features(reviews, labels,root_path, image_paths):
#     features = []
#     tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')
#     processor = ViTImageProcessor.from_pretrained('models/vit-base-patch16-224')
#
#     batch_size = 8
#     max_text_length = 128  # 确保所有文本的最大长度一致
#     img_size = (224, 224)  # ViT 模型输入的图像尺寸
#
#     processed_count = 0  # 初始化计数器
#
#     if isinstance(reviews, np.ndarray):
#         reviews = reviews.tolist()
#
#     # 统一处理所有样本的文本
#     all_text_inputs = tokenizer(reviews, padding=True, truncation=True, max_length=max_text_length,
#                                 return_tensors='pt').to(device)
#     # 分批处理图像
#     for i in range(0, len(image_paths), batch_size):
#         batch_image_paths = image_paths[i:i + batch_size]
#         batch_images = [Image.open(os.path.join(root_path, img_path)).convert("RGB").resize(img_size) for img_path in
#                         batch_image_paths]
#         batch_img_inputs = processor(images=batch_images, return_tensors="pt").to(device)
#
#         batch_texts = all_text_inputs['input_ids'][i:i + batch_size]
#         batch_masks = all_text_inputs['attention_mask'][i:i + batch_size]
#         batch_labels = labels[i:i + batch_size]
#
#         for input_id, image_input, mask, label in zip(batch_texts, batch_img_inputs['pixel_values'], batch_masks,
#                                                       batch_labels):
#             features.append(
#                 InputFeatures(
#                     text_feat=input_id.detach().cpu().numpy(),
#                     img_feat=image_input.detach().cpu().numpy(),
#                     mask_id=mask.detach().cpu().numpy(),
#                     label_id=label
#                 )
#             )
#
#         processed_count += 1
#         if processed_count % 100 == 0:
#             print(f"已处理 {processed_count * batch_size} 条数据")
#
#     return features


def MMD(source, target):
    mmd_loss = torch.exp(-1 / (source.mean(dim=0) - target.mean(dim=0)).norm())
    return mmd_loss


def multi_get_data_loader(features, batch_size):
    # 将所有特征数据转换为 numpy.ndarray 后，再转为 PyTorch 张量
    text_feats = np.array([f.text_feat for f in features], dtype=np.int64)
    mask_ids = np.array([f.mask_id for f in features], dtype=np.int64)
    img_feats = np.array([f.img_feat for f in features], dtype=np.float32)
    label_ids = np.array([f.label_id for f in features], dtype=np.int64)

    # 再将 numpy.ndarray 转为 PyTorch 张量
    all_input_text = torch.tensor(text_feats, dtype=torch.long)
    all_input_mask = torch.tensor(mask_ids, dtype=torch.long)
    all_input_img = torch.tensor(img_feats, dtype=torch.float)
    all_label_ids = torch.tensor(label_ids, dtype=torch.long)

    # 创建 TensorDataset 和 DataLoader
    dataset = TensorDataset(all_input_text, all_input_mask, all_input_img, all_label_ids)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


def roberta_convert_examples_to_features(reviews, labels, max_seq_length, tokenizer,
                                         cls_token='<s>', sep_token='</s>',
                                         pad_token=1):
    features = []
    for ex_index, (review, label) in enumerate(zip(reviews, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(reviews)))
        tokens = tokenizer.tokenize(review)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        tokens = [cls_token] + tokens + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens).to(device)
        input_mask = [1] * len(input_ids).to(device)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length).to(device)
        input_mask = input_mask + ([0] * padding_length).to(device)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          label_id=label))
    return features


def get_data_loader(features, batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
