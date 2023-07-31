from utils import api_init
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
        from utils import generate_ridge
        generate_ridge(args.json_file_dict,data_path=args.path_tar)
    if args.generate_quality:
        from EyeQualityModule import generate_quality
        generate_quality(args.path_tar)
    if args.generate_vessel:
        from VesselSegModule import generate_vessel
        generate_vessel(args.path_tar)
    if args.generate_pos_embed:
        from PositionEmbedModule import generate_pos_embed
        generate_pos_embed(args.path_tar)
    if args.generate_ridge_diffusion:
        from utils import generate_ridge_diffusion
        generate_ridge_diffusion(args.path_tar)
    if args.generate_ridge_segment:
        from ridgeSegModule import generate_ridge_segmentation
        generate_ridge_segmentation(args.path_tar)
    
    if args.generate_optic_disc:
        pass#TODO