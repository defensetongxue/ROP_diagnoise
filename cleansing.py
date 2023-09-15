from utils import api_init
from config import get_config
if __name__=='__main__':
    
    args=get_config()
    if args.generate_quality:
        from EyeQualityModule import generate_quality
        generate_quality(args.data_path,model_dict=args.model_dict)
    if args.generate_vessel:
        from VesselSegModule import generate_vessel
        generate_vessel(args.data_path,model_dict=args.model_dict)
    if args.generate_ridge_segment:
        if args.generate_pos_embed:
            from PositionEmbedModule import generate_pos_embed
            generate_pos_embed(args.data_path,model_dict=args.model_dict)
        from ridgeSegModule import generate_ridge_segmentation
        generate_ridge_segmentation(args.data_path,model_dict=args.model_dict)
    if args.generate_optic_disc:
        pass#TODO