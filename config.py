import argparse

def get_config():
    
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--path_src', type=str, default="../autodl-tmp/data_original", help='Where the data is')
    parser.add_argument('--path_tar', type=str, default="../autodl-tmp/dataset_ROP", help='Where the data generate')
    parser.add_argument('--json_file_dict', type=str, default="./json_src",
                        help='Path to the source folder containing original datasets.')
    parser.add_argument('--model_dict', type=str, default="./model_save", help='Where the data generate')
    
    parser.add_argument('--generate_ridge', type=bool, default=False, help='if parse orginal ridge annotation data')
    
    parser.add_argument('--generate_ridge_diffusion', type=bool, default=False, help='if generate ridge diffusion from ridge coordinate')
    parser.add_argument('--data_augument', type=bool, default=False, help='if doing optic disc detection')
    
    # train and test
    args = parser.parse_args()
    
    return args