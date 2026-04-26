import argparse
from semantic_uq.api import compute_uncertainty_precomputed, compute_uncertainty_onthefly

def parse_args():
    parser = argparse.ArgumentParser(description="Generative Uncertainty Calculation")
    parser.add_argument("--path", type=str, required=True, help="Path to the samples output dir (e.g. dir containing 0/imgs)")
    parser.add_argument("--mode", type=str, required=True, choices=["precomputed", "onthefly"], help="Mode of operation")
    parser.add_argument("--M", type=int, required=False, default=6, help="Number of MC samples")
    parser.add_argument(
        "--encoder",
        type=str,
        default="clip",
        choices=["clip", "dinov2_vits14_reg", "vgg16", "siglip", "openclip_h14"],
        help="Encoder used to extract image features",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of images per chunk when computing uncertainty",
    )
    parser.add_argument(
        "--entropy-calculation",
        type=str,
        default="diagonal",
        choices=["full", "diagonal", "trace"],
        help="Method for calculating Gaussian entropy: 'full', 'diagonal', or 'trace' (Total Variance)",
    )
    
    # On-the-fly specific parameters
    parser.add_argument("--sigma", type=float, default=0.02, help="Scale of the Gaussian pixel noise for onthefly mode")
    parser.add_argument("--use_transforms", action="store_true", help="Enable Mixed TTA (Crop + Flip + Noise) for onthefly mode")
    parser.add_argument(
        "--unanchored-variance",
        action="store_true",
        help="If set, uses traditional mean-centered variance instead of anchoring uncertainty strictly to the unperturbed base image.",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "precomputed":
        print(f"Running in PRECOMPUTED mode on {args.path} (anchored={not args.unanchored_variance})")
        compute_uncertainty_precomputed(
            path=args.path,
            m_samples=args.M,
            encoder_name=args.encoder,
            entropy_calculation=args.entropy_calculation,
            anchor_base=not args.unanchored_variance,
            chunk_size=args.chunk_size
        )
    elif args.mode == "onthefly":
        base_dir = args.path + "/0/imgs"
        print(f"Running in ONTHEFLY mode on base directory {base_dir} (anchored={not args.unanchored_variance})")
        chunk_size = max(args.chunk_size // args.M, 1)
        compute_uncertainty_onthefly(
            base_dir=base_dir,
            m_samples=args.M,
            noise_scale=args.sigma,
            use_transforms=args.use_transforms,
            encoder_name=args.encoder,
            entropy_calculation=args.entropy_calculation,
            anchor_base=not args.unanchored_variance,
            batch_size=chunk_size
        )
