import deepdish as dd
import numpy as np
import torch
import argparse


def rank_tensor_builder(query_f, gallery_f, topk ):
    """TODO: Docstring for rank_tensor_builder.
    :returns: TODO

    """
    dist = torch.mm(query_f, gallery_f.transpose(1, 0))
    sorted_dist, sorted_idx = torch.sort(dist, dim=1, descending=True)

    return sorted_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-path')
    parser.add_argument('--save-path')
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args()

    df = dd.io.load(args.feat_path)
    query_feat, query_label = df['query_f'], df['query_label']
    gallery_feat, gallery_label = df['gallery_f'], df['gallery_label']

    query_feat = torch.FloatTensor(query_feat)
    gallery_feat = torch.FloatTensor(gallery_feat)

    query_label = torch.LongTensor(query_label)
    gallery_label = torch.LongTensor(gallery_label)

    topk = args.top_k
    sorted_idx = rank_tensor_builder(query_feat, gallery_feat, topk)

    k_nearest_gallery = gallery_feat.repeat(query_feat.size(0), 1)[sorted_idx[:, :topk]]
    k_nearest_gallery_label = torch.gather(gallery_label.repeat(query_feat.size(0), 1),
            1, sorted_idx)[:, :topk]

    dd.io.save('global_ranks.h5',
            {
                'query': query_feat,
                'k_nearest_gallery': k_nearest_gallery,
                'query_label': query_label,
                'k_nearest_gallery_label': k_nearest_gallery_label
                })

