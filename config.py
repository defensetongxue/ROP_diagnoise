import argparse

def get_config():
    
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--data_path', type=str, default="../autodl-tmp/dataset_ROP", help='Where the data generate')
    parser.add_argument('--model_dict', type=str, default="./model_save", help='Where the data generate')
    parser.add_argument('--generate_vessel', type=bool, default=True, help='if parse orginal ridge annotation data')
    parser.add_argument('--generate_quality', type=bool, default=False, help='if parse orginal ridge annotation data')
    parser.add_argument('--generate_ridge_segment', type=bool, default=False, help='if parse orginal ridge annotation data')
    parser.add_argument('--generate_optic_disc', type=bool, default=False, help='if parse orginal ridge annotation data')
    # train and test
    args = parser.parse_args()
    
    return args