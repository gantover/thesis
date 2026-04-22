import click
from scoring.fid import fid
from scoring.idx_sort import idx_sort
from scoring.precision_recall_torch import precision_recall
import click
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import torch
import torch_utils.distributed as dist

@click.command()
@click.option('--exp_path')
@click.option('--ref_path')
def main(exp_path, ref_path):
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    if not torch.distributed.is_initialized():
        dist.init()

    scores = []
    H = 128
    fid_images_path = str(Path(exp_path) / "0" / "imgs")
    fid_refs_path = str(Path(ref_path) / "fid-refs" / f"imagenet-{H}x{H}.npz")
    prt_ref_path = str(Path(ref_path) / "precision-recall-refs" / f"image_net_val_{H}_fid_features_.pt")
    #--ref $REF_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt 
    # --eval "${EXP_PATH}/${m}/fid_features.pt"
    for N in [12000, 11000, 10000, 9000, 8000, 7000, 6000]:
    # for N in tqdm([1000, 900, 800, 700, 600, 500]):
        # random baseline
        fid_features_path = str(Path(exp_path) / "0" / "fid_features.pt")
        random_fid = fid(
            image_path=fid_images_path,
            ref_path=fid_refs_path,
            num_expected=N,
            fid_features=fid_features_path,
            )
        prt_eval_features_path = str(Path(exp_path) / "0" / "fid_features.pt")
        random_precision, random_recall = precision_recall(
            ref_features_path=prt_ref_path,
            eval_features_path=prt_eval_features_path,
            row_batch_size=2000,
            col_batch_size=2000
        )
        print(f"FID: {random_fid} Precision: {random_precision}, Recall: {random_recall}")
        scores.append({"N": N, "Method": "Random", "FID": random_fid, "Precision": random_precision, "Recall": random_recall})
        
        torch.cuda.empty_cache()

        # G.U. baseline
        idx_sort(path=exp_path, name="entropy_clip", N=N, reverse=False)
        gu_fid = fid(
            image_path = fid_images_path,
            ref_path = fid_refs_path,
            num_expected = 50000,
            fid_features = str(Path(exp_path) / "0" / "fid_features_filtered_entropy_clip.pt"),
            idx_path = str(Path(exp_path) / f"idx_sorted_{N}_entropy_clip.npy")
        )
        gu_precision, gu_recall = precision_recall(
            ref_features_path=prt_ref_path,
            eval_features_path=str(Path(exp_path) / "0" / "fid_features_filtered_entropy_clip.pt"),
            row_batch_size=2000,
            col_batch_size=2000
        )
        print(f"FID: {gu_fid} Precision: {gu_precision}, Recall: {gu_recall}")
        scores.append({"N": N, "Method": "G.U.", "FID": gu_fid, "Precision": gu_precision, "Recall": gu_recall})

        idx_sort_path = str(Path(exp_path) / "0")
        
        torch.cuda.empty_cache()
        # realism baseline
        idx_sort(path=idx_sort_path, name="realism", N=N, reverse=True)
        realism_fid = fid(
            image_path = fid_images_path,
            ref_path = fid_refs_path,
            num_expected = 50000,
            fid_features = str(Path(exp_path) / "0" / "fid_features_filtered_realism.pt"),
            idx_path = str(Path(exp_path) / "0" / f"idx_sorted_{N}_realism.npy")
        )
        realism_precision, realism_recall = precision_recall(
            ref_features_path=prt_ref_path,
            eval_features_path=str(Path(exp_path) / "0" / "fid_features_filtered_realism.pt"),
            row_batch_size=2000,
            col_batch_size=2000
        )
        print(f"FID: {realism_fid} Precision: {realism_precision}, Recall: {realism_recall}")
        scores.append({"N": N, "Method": "Realism", "FID": realism_fid, "Precision": realism_precision, "Recall": realism_recall})
        
        torch.cuda.empty_cache()
        # rarity baseline
        idx_sort(path=idx_sort_path, name="rarity", N=N, reverse=False)
        rarity_fid = fid(
            image_path = fid_images_path,
            ref_path = fid_refs_path,
            num_expected = 50000,
            fid_features = str(Path(exp_path) / "0" / "fid_features_filtered_rarity.pt"),
            idx_path = str(Path(exp_path) / "0" / f"idx_sorted_{N}_rarity.npy")
        )
        rarity_precision, rarity_recall = precision_recall(
            ref_features_path=prt_ref_path,
            eval_features_path=str(Path(exp_path) / "0" / "fid_features_filtered_rarity.pt"),
            row_batch_size=2000,
            col_batch_size=2000
        )
        print(f"FID: {rarity_fid} Precision: {rarity_precision}, Recall: {rarity_recall}")
        scores.append({"N": N, "Method": "Rarity", "FID": rarity_fid, "Precision": rarity_precision, "Recall": rarity_recall})

        torch.cuda.empty_cache()

        scores_df = pd.DataFrame(scores)
        scores_df.to_pickle(Path(exp_path) / "scores.pkl")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
