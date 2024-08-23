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
from PIL import Image
from torchvision import models, transforms
from cross_attention import CrossAttention  # 导入 CrossAttention 类

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
        return 2
    elif label == 'neutral':
        return 1
    elif label == 'negative':
        return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, text_feat, img_feat, label_id):
        self.text_feat = text_feat
        self.img_feat = img_feat
        self.label_id = label_id


def XML2Array(neg_path, pos_path):
    parser = etree.XMLParser(recover=True)
    reviews = []
    negCount = 0
    posCount = 0
    labels = []
    regex = re.compile(r'[\n\r\t+]')

    neg_tree = ET.parse(neg_path, parser=parser)
    neg_root = neg_tree.getroot()

    for rev in neg_root.iter('review_text'):
        text = regex.sub(" ", rev.text)
        reviews.append(text)
        negCount += 1
    labels.extend(np.zeros(negCount, dtype=int))

    pos_tree = ET.parse(pos_path, parser=parser)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review_text'):
        text = regex.sub(" ", rev.text)
        reviews.append(text)
        posCount += 1
    labels.extend(np.ones(posCount, dtype=int))

    reviews = np.array(reviews)
    labels = np.array(labels)

    return reviews, labels


def CSV2Array(path):
    data = pd.read_csv(path, encoding='latin')
    reviews, labels = data.reviews.values.tolist(), data.labels.values.tolist()
    return reviews, labels


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
    texts = []
    labels = []
    # 打开文件并逐行读取
    with open(path, 'r', encoding='utf-8') as file:
        # 跳过第一行（列名）
        next(file)
        for line in file:
            # 去除行尾的换行符和空白
            line = line.strip()
            # 分割每行数据
            parts = line.split('\t')
            # 读取第二列和第三列
            text = parts[1]
            label = convert_label(parts[2])
            # 添加到列表中
            texts.append(text)
            labels.append(label)
    return texts, labels


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
                          input_mask=input_mask,
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


def multi_convert_examples_to_features(reviews, labels, root_path, image_paths):
    features = []
    tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')
    text_model = BertModel.from_pretrained("models/bert-base-uncased").to(device)
    processor = ViTImageProcessor.from_pretrained('models/vit-base-patch16-224')
    img_model = ViTModel.from_pretrained('models/vit-base-patch16-224').to(device)

    batch_size = 8

    for i in range(0, len(reviews), batch_size):
        batch_texts = reviews[i:i + batch_size]
        batch_images = image_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        if isinstance(batch_texts, np.ndarray):
            batch_texts = batch_texts.tolist()

        print(f"Type of batch_texts: {type(batch_texts)}")
        print(f"Contents of batch_texts: {batch_texts}")

        processed_images = []
        for image_path in batch_images:
            full_path = os.path.join(root_path, image_path)
            image = Image.open(full_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_images.append(image)

        # 处理文本和图像输入
        text_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        text_outputs = text_model(**text_inputs)

        # 处理后的 processed_images 列表可以用于后续的图像处理
        img_inputs = processor(images=processed_images, return_tensors="pt").to(device)
        image_outputs = img_model(**img_inputs)


        # 提取[CLS] token的特征表示
        text_cls_embeddings = text_outputs.last_hidden_state[:, 0, :]
        image_cls_embeddings = image_outputs.last_hidden_state[:, 0, :]

        # 将每个样本的特征和标签保存到特征数组中
        for text_cls, image_cls, label in zip(text_cls_embeddings, image_cls_embeddings, batch_labels):
            features.append(
                InputFeatures(
                    text_feat=text_cls.detach().cpu().numpy(),  # 将 tensor 转换为 numpy 数组
                    img_feat=image_cls.detach().cpu().numpy(),  # 将 tensor 转换为 numpy 数组
                    label_id=label
                )
            )

        # 清理缓存，释放显存
        torch.cuda.empty_cache()



    return features


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


def mulit_get_data_loader(features, batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    # 使用 clone().detach() 直接复制张量
    all_img_datas = torch.stack([f.img_datas.clone().detach().float() for f in features])

    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_img_datas)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def get_data_loader(features, batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def MMD(source, target):
    mmd_loss = torch.exp(-1 / (source.mean(dim=0) - target.mean(dim=0)).norm())
    return mmd_loss

# 多模态数据
