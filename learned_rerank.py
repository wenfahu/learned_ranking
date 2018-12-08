import torch
import pdb

BS = 64

def batchify2(iterable, n):
    """TODO: Docstring for batchify2.
    :returns: TODO

    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx+n, l)]

def re_rank(query_feat, gallery_feat, model, topk=50):
    """TODO: Docstring for re_rank.
    :returns: TODO

    """
    dist = torch.mm(query_feat, gallery_feat.transpose(1, 0))
    k_dist, k_idx = torch.topk(dist, k=topk, dim=1 )
    sorted_dsit, sorted_idx = torch.sort(dist, dim=1, descending=True)
    # k_idx of shape (query_size x k)

    # gallery_feat of shape GALLARY_SIZE x FEAT_DIM

    # QUERY_SIZE x GALLARY_SIZE x FEAT_DIM

    k_nearest_gallery = [ gallery_feat[idx,:].unsqueeze_(0) for idx in k_idx]
    k_nearest_gallery = torch.cat(k_nearest_gallery)
    # k_nearest_gallery = torch.gather(gallery_feat.repeat(query_feat.size(0), 1),
    #         dim=1, index=k_idx)

    # QUERY_SIZE x K+1 x FEAT_DIM
    query_gallery = torch.cat((query_feat.unsqueeze(1), k_nearest_gallery), dim=1)

    # relations = (k_nearest_gallery.unsqueeze_(2) - query_gallery.unsqueeze_(1)) ** 2
    # relations = relations.permute(0, 3, 1, 2)

    score = torch.zeros(query_feat.size(0), topk)
    idx = 0
    for kn_batch, qg_batch in zip(*[batchify2(p, BS) for p in 
        [k_nearest_gallery, query_gallery]]):

        relations = (kn_batch.unsqueeze_(2) - qg_batch.unsqueeze_(1)) ** 2

        score[idx*BS: min(BS*(idx+1), score.size(0)), : ]= model(
                relations.permute(0, 3, 1, 2).cuda()).data.cpu()
        idx = idx + 1
                

    sorted_score, re_arange = torch.sort(score, dim=1, descending=True)
    sorted_idx[:, :topk] = torch.gather(sorted_idx[:, :topk], 1, re_arange)
    import pdb
    pdb.set_trace()
    # rescored_dist = torch.gather(dist, 1, k_idx) * score
    # results.scatter_(1, k_idx, rescored_dist)

    return sorted_idx





