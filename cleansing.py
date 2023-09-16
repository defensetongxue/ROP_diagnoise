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
    if args.generate_pos_embed:
        from PositionEmbedModule import generate_pos_embed
        generate_pos_embed(args.data_path,model_dict=args.model_dict,split_name=args.split_name)
    if args.generate_ridge_segment:
        from ridgeSegModule import generate_ridge_segmentation
        generate_ridge_segmentation(args.data_path,model_dict=args.model_dict,
                                    ridge_seg_number=args.ridge_seg_number,
                                    ridge_seg_dis=args.ridge_seg_distance)
    if args.generate_optic_disc:
        from OpticDetectModule import generate_optic_disc_location
        generate_optic_disc_location(args.data_path,
                                     split_name=args.split_name,
                                     model_dict=args.model_dict)