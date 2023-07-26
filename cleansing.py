from utils_ import api_init
from config import get_config
if __name__=='__main__':
    
    args=get_config()
    api_init(data_path=args.path_tar,content={
        'image_path':"path to image",
        "id":"image id",
        "Stage":"Stage for ROP",
        "Zone":"Zone for ROP",
        "Plus":"pre plus or plus ROP, 0 normal 1 pre plus 2 plus",
    })
    
    if args.generate_ridge:
        from utils_ import generate_ridge
        generate_ridge(args.json_file_dict,data_path=args.path_tar)
    if args.generate_quality:
        pass
    if args.generate_vessel:
        pass
    if args.generate_pos_embed:
        pass
    if args.generate_ridge_diffusion:
        from utils_ import generate_ridge_diffusion
        generate_ridge_diffusion(args.path_tar)
    if args.generate_ridge_segment:
        pass
    if args.generate_optic_disc:
        pass