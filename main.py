"""Main script for ADDA."""

import param
from train import pretrain, adapt, evaluate, multi_pretrain, multi_evaluate, multi_adapt
from model import (BertEncoder, DistilBertEncoder, DistilRobertaEncoder,
                   BertClassifier, Discriminator, RobertaEncoder, RobertaClassifier, ViTEncoder)
from utils import convert_examples_to_features, \
    roberta_convert_examples_to_features, get_data_loader, init_model, init_multi_model, TWI_CSV2Array, \
    multi_convert_examples_to_features, mulit_get_data_loader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, RobertaTokenizer
import torch
import os
import random
import argparse
import numpy as np

path = "datasets/IJCAI2019_data/twitter2015"


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    # parser.add_argument('--src', type=str, default="books",
    #                     choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb"],
    #                     help="Specify src dataset")
    #
    # parser.add_argument('--tgt', type=str, default="dvd",
    #                     choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb"],
    #                     help="Specify tgt dataset")
    parser.add_argument('--src', type=str, default="IJCAI2019_data/twitter2015",
                        choices=["IJCAI2019_data/twitter2015", "IJCAI2019_data/twitter2017"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="IJCAI2019_data/twitter2017",
                        choices=["IJCAI2019_data/twitter2015", "IJCAI2019_data/twitter2017"],
                        help="Specify tgt dataset")

    parser.add_argument('--pretrain', default=True, action='store_true',
                        help='Force to pretrain source encoder/classifier')

    parser.add_argument('--adapt', default=True, action='store_true',
                        help='Force to adapt target encoder')
    # TRUE 43:0.9075
    # FALSE 43:0.5727
    parser.add_argument('--seed', type=int, default=43,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=43,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="bert-base-uncased",
                        choices=["bert-base-uncased", "distilbert", "roberta", "distilroberta"],
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Specify adversarial weight")

    parser.add_argument('--beta', type=float, default=1.0,
                        help="Specify KD loss weight")

    parser.add_argument('--temperature', type=int, default=20,
                        help="Specify temperature")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")

    parser.add_argument('--batch_size', type=int, default=16,
                        help="Specify batch size")

    parser.add_argument('--pre_epochs', type=int, default=3,
                        help="Specify the number of epochs for pretrain")

    parser.add_argument('--pre_log_step', type=int, default=1,
                        help="Specify log step size for pretrain")

    parser.add_argument('--num_epochs', type=int, default=3,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")

    parser.add_argument('--multimodal', default=True, action='store_true',
                        help='Use multimodal data')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_arguments()
    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("seed: " + str(args.seed))
    print("train_seed: " + str(args.train_seed))
    print("model_type: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("pre_epochs: " + str(args.pre_epochs))
    print("num_epochs: " + str(args.num_epochs))
    print("AD weight: " + str(args.alpha))
    print("KD weight: " + str(args.beta))
    print("temperature: " + str(args.temperature))
    set_seed(args.train_seed)

    if args.model in ['roberta', 'distilroberta']:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        tokenizer = BertTokenizer.from_pretrained('./models/bert-base-uncased')

    # preprocess data
    print("=== Processing datasets ===")
    # if args.src in ['blog', 'airline', 'imdb']:
    #     src_x, src_y = CSV2Array(os.path.join('data', args.src, args.src + '.csv'))
    # else:
    #     src_x, src_y = XML2Array(os.path.join('data', args.src, 'negative.review'),
    #                            os.path.join('data', args.src, 'positive.review'))
    print(args.src)
    if args.src in ["IJCAI2019_data/twitter2015", "IJCAI2019_data/twitter2017"]:
        src_label, src_image, src_string = TWI_CSV2Array(os.path.join('datasets', args.src, 'dev.tsv'),
                                                         os.path.join('datasets', args.src, 'test.tsv'),
                                                         os.path.join('datasets', args.src, 'train.tsv')
                                                         )
    if args.src in ["mvsa"]:
        src_label, src_image, src_string = [], [], []
        print("0000")

    src_train_label, src_test_label, src_train_image, src_test_image, src_train_string, src_test_string = train_test_split(
        src_label,
        src_image,
        src_string,
        test_size=0.2,
        stratify=src_label,
        random_state=args.seed)

    if args.tgt in ["IJCAI2019_data/twitter2015", "IJCAI2019_data/twitter2017"]:
        tgt_label, tgt_image, tgt_string = TWI_CSV2Array(os.path.join('datasets', args.tgt, 'dev.tsv'),
                                                         os.path.join('datasets', args.tgt, 'test.tsv'),
                                                         os.path.join('datasets', args.tgt, 'train.tsv')
                                                         )
    else:
        tgt_label, tgt_image, tgt_string = [], [], []
        print("0000")

    tgt_train_label, tgt_test_label, tgt_train_image, tgt_test_image, tgt_train_string, tgt_test_string = train_test_split(
        tgt_label,
        tgt_image,
        tgt_string,
        test_size=0.2,
        stratify=tgt_label,
        random_state=args.seed)

    if args.model in ['roberta', 'distilroberta']:
        src_features = roberta_convert_examples_to_features(src_string, src_label, args.max_seq_length, tokenizer)
        src_test_features = roberta_convert_examples_to_features(src_test_string, src_test_label, args.max_seq_length,
                                                                 tokenizer)
        tgt_features = roberta_convert_examples_to_features(tgt_string, tgt_label, args.max_seq_length, tokenizer)
        tgt_train_features = roberta_convert_examples_to_features(tgt_string, tgt_label,
                                                                  args.max_seq_length,
                                                                  tokenizer)
        src_root_path = os.path.join('datasets', args.src + "_images")
        if args.multimodal:
            # test*********
            print(args.multimodal)

    else:

        if args.multimodal:
            src_root_path = os.path.join('datasets', args.src + "_images")
            src_features = multi_convert_examples_to_features(src_string, src_label, src_root_path, src_image)

            src_test_features = multi_convert_examples_to_features(src_test_string, src_test_label, src_root_path,
                                                                   src_test_image)

            tgt_root_path = os.path.join('datasets', args.tgt + "_images")
            print(tgt_root_path)
            tgt_features = multi_convert_examples_to_features(tgt_string, tgt_label,
                                                              tgt_root_path, tgt_image)
            tgt_train_features = multi_convert_examples_to_features(tgt_train_string, tgt_train_label,
                                                                    tgt_root_path, tgt_train_image)
            print("**************")

        else:
            src_features = convert_examples_to_features(src_string, src_label, args.max_seq_length,
                                                        tokenizer)
            src_test_features = convert_examples_to_features(src_test_string, src_test_label, args.max_seq_length,
                                                             tokenizer)
            tgt_features = convert_examples_to_features(tgt_string, tgt_label, args.max_seq_length, tokenizer)
            tgt_train_features = convert_examples_to_features(tgt_train_string, tgt_train_label, args.max_seq_length,
                                                              tokenizer)

    # load dataset
    if args.multimodal:
        src_data_loader = mulit_get_data_loader(src_features, args.batch_size)
        src_data_eval_loader = mulit_get_data_loader(src_test_features, args.batch_size)
        tgt_data_train_loader = mulit_get_data_loader(tgt_train_features, args.batch_size)
        tgt_data_all_loader = mulit_get_data_loader(tgt_features, args.batch_size)
    else:
        src_data_loader = get_data_loader(src_features, args.batch_size)
        src_data_eval_loader = get_data_loader(src_test_features, args.batch_size)
        tgt_data_train_loader = get_data_loader(tgt_train_features, args.batch_size)
        tgt_data_all_loader = get_data_loader(tgt_features, args.batch_size)

    # load models
    if args.multimodal:
        if args.model == 'bert-base-uncased':
            src_encoder_t = BertEncoder()
            tgt_encoder_t = BertEncoder()
            src_encoder_i = ViTEncoder()
            tgt_encoder_i = ViTEncoder()
            src_classifier = BertClassifier()
        elif args.model == 'distilbert':
            src_encoder = DistilBertEncoder()
            tgt_encoder = DistilBertEncoder()
            src_classifier = BertClassifier()
        elif args.model == 'roberta':
            src_encoder = RobertaEncoder()
            tgt_encoder = RobertaEncoder()
            src_classifier = RobertaClassifier()
        else:
            src_encoder = DistilRobertaEncoder()
            tgt_encoder = DistilRobertaEncoder()
            src_classifier = RobertaClassifier()
        discriminator = Discriminator()
    else:
        # load models
        if args.model == 'bert':
            src_encoder = BertEncoder()
            tgt_encoder = BertEncoder()
            src_classifier = BertClassifier()
        elif args.model == 'distilbert':
            src_encoder = DistilBertEncoder()
            tgt_encoder = DistilBertEncoder()
            src_classifier = BertClassifier()
        elif args.model == 'roberta':
            src_encoder = RobertaEncoder()
            tgt_encoder = RobertaEncoder()
            src_classifier = RobertaClassifier()
        else:
            src_encoder = DistilRobertaEncoder()
            tgt_encoder = DistilRobertaEncoder()
            src_classifier = RobertaClassifier()
        discriminator = Discriminator()

    if args.multimodal:
        if args.load:
            src_encoder_t, src_encoder_i = init_multi_model(args, src_encoder_t, src_encoder_i,
                                                            restore_t=param.src_encoder_path_t,
                                                            restore_i=param.src_encoder_path_i)
            src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path)
            tgt_encoder_t, tgt_encoder_i = init_multi_model(args, tgt_encoder_t, tgt_encoder_i,
                                                            restore_t=param.src_encoder_path_t,
                                                            restore_i=param.src_encoder_path_i)
            discriminator = init_model(args, discriminator, restore=param.d_model_path)
        else:
            src_encoder_t, src_encoder_i = init_multi_model(args, src_encoder_t, src_encoder_i)
            src_classifier = init_model(args, src_classifier)
            tgt_encoder_t, tgt_encoder_i = init_multi_model(args, src_encoder_t, tgt_encoder_i)
            discriminator = init_model(args, discriminator)
    else:
        if args.load:
            src_encoder = init_model(args, src_encoder, restore=param.src_encoder_path)
            src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path)
            tgt_encoder = init_model(args, tgt_encoder, restore=param.tgt_encoder_path)
            discriminator = init_model(args, discriminator, restore=param.d_model_path)
        else:
            src_encoder = init_model(args, src_encoder)
            src_classifier = init_model(args, src_classifier)
            tgt_encoder = init_model(args, tgt_encoder)
            discriminator = init_model(args, discriminator)

    # train source model
    print("=== Training classifier for source domain ===")
    if args.pretrain:
        if args.multimodal:
            src_encoder_t, src_encoder_i, src_classifier = multi_pretrain(
                args, src_encoder_t, src_encoder_i, src_classifier, src_data_loader)
        else:
            src_encoder, src_classifier = pretrain(
                args, src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    if args.multimodal:
        multi_evaluate(src_encoder_t, src_encoder_i, src_classifier, src_data_loader)
        multi_evaluate(src_encoder_t, src_encoder_i, src_classifier, src_data_eval_loader)
        multi_evaluate(src_encoder_t, src_encoder_i, src_classifier, tgt_data_all_loader)
    else:
        evaluate(src_encoder, src_classifier, src_data_loader)
        evaluate(src_encoder, src_classifier, src_data_eval_loader)
        evaluate(src_encoder, src_classifier, tgt_data_all_loader)

    for params in src_encoder_t.parameters():
        params.requires_grad = False
    for params in src_encoder_i.parameters():
        params.requires_grad = False

    for params in src_classifier.parameters():
        params.requires_grad = False

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    if args.adapt:
        if args.multimodal:
            tgt_encoder.load_state_dict(src_encoder.state_dict())
            tgt_encoder = multi_adapt(args, src_encoder, tgt_encoder, discriminator,
                                      src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_all_loader)

        else:
            tgt_encoder.load_state_dict(src_encoder.state_dict())
            tgt_encoder = adapt(args, src_encoder, tgt_encoder, discriminator,
                                src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_all_loader)

    # eval target encoder on lambda0.1 set of target dataset
    if args.multimodal:
        print("=== Evaluating classifier for encoded target domain ===")
        print(">>> source only <<<")
        multi_evaluate(src_encoder, src_classifier, tgt_data_all_loader)
        print(">>> domain adaption <<<")
        multi_evaluate(tgt_encoder, src_classifier, tgt_data_all_loader)
    else:
        print("=== Evaluating classifier for encoded target domain ===")
        print(">>> source only <<<")
        evaluate(src_encoder, src_classifier, tgt_data_all_loader)
        print(">>> domain adaption <<<")
        evaluate(tgt_encoder, src_classifier, tgt_data_all_loader)


if __name__ == '__main__':
    main()
