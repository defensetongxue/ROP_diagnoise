import argparse
import yaml
from yacs.config import CfgNode as CN

_C = CN()

_C.RESULT_PATH = './experiments'

_C.MODEL = CN()
_C.MODEL.MODEL_NAME = 'inceptionv3'
_C.MODEL.SAVE_DIR = './checkpoints'
_C.MODEL.SAVE_NAME = 'best.pth'

_C.TRAIN = CN()
_C.TRAIN.BEGIN_CHECKPOINT = ''
_C.TRAIN.BATCH_SIZE_PER_GPU = 128
_C.TRAIN.NUM_WORKERS = 12
_C.TRAIN.SHUFFLE = True
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 500
_C.TRAIN.EARLY_STOP = 30
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.LR = 0.0001
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.WD = 0.0
_C.TRAIN.LR_STEP = [30, 50]
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.NESTEROV = False

def update_config(cfg: CN, args) -> None:
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

def get_config():
    
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--path_src', type=str, default="../autodl-tmp/data_original", help='Where the data is')
    parser.add_argument('--path_tar', type=str, default="../autodl-tmp/dataset_ROP", help='Where the data generate')
    parser.add_argument('--train_split', type=float, default=0.5, help='training data proportion')
    parser.add_argument('--val_split', type=float, default=0.25, help='valid data proportion')

    parser.add_argument('--cleansing', type=bool, default=True, help='if parse orginal data')
    parser.add_argument('--data_augument', type=bool, default=True, help='if doing optic disc detection')

    # train and test
    parser.add_argument('--config_file', type=str, default='./YAML/default.yaml', help='load config file')

    args = parser.parse_args()
    
    return args