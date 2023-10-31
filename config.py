import argparse

def get_config():
    
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--data_path', type=str, default="../autodl-tmp/dataset_ROP", help='Where the data generate')
    parser.add_argument('--split_name', type=str, default="1", help='Where the data generate')
    parser.add_argument('--model_dict', type=str, default="./model_save", help='Where the data generate')
    parser.add_argument('--generate_vessel', type=bool, default=False, help='if parse orginal ridge annotation data')
    parser.add_argument('--generate_quality', type=bool, default=False, help='if parse orginal ridge annotation data')
    parser.add_argument('--generate_pos_embed', type=bool, default=False, help='if parse orginal ridge annotation data')
    parser.add_argument('--generate_ridge_segment', type=bool, default=False, help='if parse orginal ridge annotation data')
    parser.add_argument('--ridge_seg_number', type=int, default=5, help='if parse orginal ridge annotation data')
    parser.add_argument('--ridge_seg_distance', type=int, default=150, help='if parse orginal ridge annotation data')
    parser.add_argument('--generate_optic_disc', type=bool, default=True, help='if parse orginal ridge annotation data')
    # train and test
    args = parser.parse_args()
    
    return args