import click
from scoring.fid import calculate_fid_from_inception_stats
from scoring.idx_sort import idx_sort
from scoring.precision_recall_torch import knn_precision_recall_features, compute_rarity_scores

import click
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import dnnlib

def get_stats_from_features(features):
    features = features.to(torch.float64)
    N = features.shape[0]
    mu = features.sum(dim=0)
    sigma = features.T @ features
    mu /= N
    sigma -= mu.ger(mu) * N
    sigma /= N - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

@click.command()
@click.option('--exp_path')
@click.option('--ref_path')
def main(exp_path, ref_path):
    H = 128
    
    fid_refs_path = str(Path(ref_path) / "fid-refs" / f"imagenet-{H}x{H}.npz")
    prt_ref_path = str(Path(ref_path) / "precision-recall-refs" / f"image_net_val_{H}_fid_features_.pt")
    
    print(f"Loading reference FID stats from {fid_refs_path}")
    with dnnlib.util.open_url(fid_refs_path) as f:
        ref_stats = dict(np.load(f))
        ref_mu = ref_stats["mu"]
        ref_sigma = ref_stats["sigma"]
    
    print(f"Loading reference PRT features from {prt_ref_path}")
    prt_ref_features = torch.load(prt_ref_path).cpu().numpy()
    
    print(f"Loading generated features")
    # All generated features
    full_eval_features_pt = torch.load(str(Path(exp_path) / "0" / "fid_features_all.pt")).cpu()
    
    scores = []

    for N in [12000, 11000, 10000, 9000, 8000, 7000, 6000]:
        print(f"Processing N={N}")
        
        # 1. Random baseline
        # print("  - Random")
        # Use first N features
        # eval_features_pt = full_eval_features_pt[:N]
        # mu, sigma = get_stats_from_features(eval_features_pt)

        # random_fid = calculate_fid_from_inception_stats(mu, sigma, ref_mu, ref_sigma)
        # random_pr_dict = knn_precision_recall_features(
        #     prt_ref_features, 
        #     eval_features_pt.numpy(), 
        #     row_batch_size=10000, col_batch_size=10000
        # )
        # random_precision, random_recall = random_pr_dict["precision"], random_pr_dict["recall"]
        # print(f"    FID: {random_fid} Precision: {random_precision}, Recall: {random_recall}")
        # scores.append({"N": N, "Method": "Random", "FID": random_fid, "Precision": random_precision, "Recall": random_recall})
        
        # 2. G.U. baseline
        print("  - G.U.")
        idx_sort(path=exp_path, name="entropy_clip", N=N, reverse=False)
        idx_gu = np.load(str(Path(exp_path) / f"idx_sorted_{N}_entropy_clip.npy"))
        eval_features_pt = full_eval_features_pt[idx_gu]
        
        mu, sigma = get_stats_from_features(eval_features_pt)
        gu_fid = calculate_fid_from_inception_stats(mu, sigma, ref_mu, ref_sigma)
        
        gu_pr_dict = knn_precision_recall_features(
            prt_ref_features, 
            eval_features_pt.numpy(), 
            row_batch_size=10000, col_batch_size=10000
        )
        gu_precision, gu_recall = gu_pr_dict["precision"], gu_pr_dict["recall"]
        print(f"    FID: {gu_fid} Precision: {gu_precision}, Recall: {gu_recall}")
        scores.append({"N": N, "Method": "G.U.", "FID": gu_fid, "Precision": gu_precision, "Recall": gu_recall})

        # 3. Realism baseline
        # print("  - Realism")
        # idx_sort_path = str(Path(exp_path) / "0")
        # idx_sort(path=idx_sort_path, name="realism", N=N, reverse=True)
        # idx_realism = np.load(str(Path(exp_path) / "0" / f"idx_sorted_{N}_realism.npy"))
        # eval_features_pt = full_eval_features_pt[idx_realism]
        
        # mu, sigma = get_stats_from_features(eval_features_pt)
        # realism_fid = calculate_fid_from_inception_stats(mu, sigma, ref_mu, ref_sigma)
        
        # realism_pr_dict = knn_precision_recall_features(
        #     prt_ref_features, 
        #     eval_features_pt.numpy(), 
        #     row_batch_size=10000, col_batch_size=10000
        # )
        # realism_precision, realism_recall = realism_pr_dict["precision"], realism_pr_dict["recall"]
        # print(f"    FID: {realism_fid} Precision: {realism_precision}, Recall: {realism_recall}")
        # scores.append({"N": N, "Method": "Realism", "FID": realism_fid, "Precision": realism_precision, "Recall": realism_recall})

        # 4. Rarity baseline
        # print("  - Rarity")
        # idx_sort(path=idx_sort_path, name="rarity", N=N, reverse=False)
        # idx_rarity = np.load(str(Path(exp_path) / "0" / f"idx_sorted_{N}_rarity.npy"))
        # eval_features_pt = full_eval_features_pt[idx_rarity]
        
        # mu, sigma = get_stats_from_features(eval_features_pt)
        # rarity_fid = calculate_fid_from_inception_stats(mu, sigma, ref_mu, ref_sigma)
        
        # rarity_pr_dict = knn_precision_recall_features(
        #     prt_ref_features, 
        #     eval_features_pt.numpy(), 
        #     row_batch_size=10000, col_batch_size=10000
        # )
        # rarity_precision, rarity_recall = rarity_pr_dict["precision"], rarity_pr_dict["recall"]
        # print(f"    FID: {rarity_fid} Precision: {rarity_precision}, Recall: {rarity_recall}")
        # scores.append({"N": N, "Method": "Rarity", "FID": rarity_fid, "Precision": rarity_precision, "Recall": rarity_recall})

        scores_df = pd.DataFrame(scores)
        scores_df.to_pickle(Path(exp_path) / "scores.pkl")

if __name__ == "__main__":
    main()
