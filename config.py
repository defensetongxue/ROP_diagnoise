import argparse

def get_config():
    
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--path_src', type=str, default="../autodl-tmp/data_original", help='Where the data is')
    parser.add_argument('--path_tar', type=str, default="../autodl-tmp/dataset_ROP", help='Where the data generate')
    parser.add_argument('--train_split', type=float, default=0.5, help='training data proportion')
    parser.add_argument('--val_split', type=float, default=0.25, help='valid data proportion')
    parser.add_argument('--json_file_dict', type=str, default="./json_src",
                        help='Path to the source folder containing original datasets.')
    parser.add_argument('--cleansing', type=bool, default=True, help='if parse orginal data')
    parser.add_argument('--generate_ridge', type=bool, default=True, help='if parse orginal data')
    parser.add_argument('--data_augument', type=bool, default=False, help='if doing optic disc detection')

    # train and test
    args = parser.parse_args()
    
    return args