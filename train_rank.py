import deepdish as dd
import torch
import argparse
import torch.nn as nn
from torch.utils.data import TensorDataset
from ScoreModel import ScoreNetwork
from batch_sampler import ClassBalancedSampler
import os


def train(gallery_loader, query_loader, model, criterion, optimizer, scheduler, topk):
    """TODO: Docstring for main.

    :arg1: TODO
    :returns: TODO

    """
    num_episodes = len(query_loader)
    for idx in range(num_episodes):
        query_f, query_l = query_loader.__iter__().next()
        gallery_f, gallery_l = gallery_loader.__iter__().next()

        dist = torch.mm(query_f, gallery_f.transpose(1, 0))
        sorted_dist, sorted_idx = torch.sort(dist, dim=1)

        k_nearest_gallery = gallery_f.repeat(query_f.size(0), 1)[sorted_idx[:, :topk]]
        # k_nearest_gallery of shape QueryBatch x TopK x FeatDim
        k_nearest_gallery_label = torch.gather(gallery_l.repeat(query_f.size(0), 1), 
                1, sorted_idx)[:, :topk]

        query_gallery = torch.cat((query_f.unsqueeze(1), k_nearest_gallery), dim=1)
        # query_gallery of shape QueryBatch x (TopK +1 ) x FeatDim
        relations = (k_nearest_gallery.unsqueeze(2) - query_gallery.unsqueeze(1)) ** 2
        # relations of shape QueryBatch x K x (K+1) x FeatDim

        import pdb
        pdb.set_trace()
        relations_label = query_l.unsqueeze(1) == k_nearest_gallery_label
        # relation label of shape QueryBatch x K

        scheduler.step()

        optimizer.zero_grad()
        out = model(relations)
        loss = criterion(out, relations_label.float())

        loss.backword()

        optimizer.step()

        if idx % 50 == 49:
            print("\tTraining Episode idx, loss {:.3f}".format(loss.item()))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-path')
    parser.add_argument('--num-classes', type=int, default=50)
    parser.add_argument('--instance-per-class', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=60)
    parser.add_argument('--train-root', default='./ranks')
    args = parser.parse_args()

    feats = dd.io.load(args.feat_path)

    gallery_feat = torch.FloatTensor(feats['gallery_f'])
    gallery_label = torch.LongTensor(feats['gallery_label'])

    query_feat = torch.FloatTensor(feats['query_f'])
    query_label = torch.LongTensor(feats['query_label'])

    gallery_set = TensorDataset(gallery_feat, gallery_label)
    query_set = TensorDataset(query_feat, query_label)

    gallery_sampler = ClassBalancedSampler(feats['gallery_label'], args.num_classes, 
            args.instace_per_class, args.episodes)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, 
            batch_sampler=gallery_sampler)
    query_sampler = ClassBalancedSampler(feats['query_label'], args.num_classes,
            1, args.episodes)
    query_loader = torch.utils.data.DataLoader(query_set, 
            batch_sampler=query_sampler)

    model = ScoreNetwork(args.topk)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.lr_decay)

    if not os.path.isdir(args.train_root):
        os.makedirs(args.train_root)

    for epoch_idx in range(args.num_epoch):
        print("Training Epoch {}".format(epoch_idx))
        train(gallery_loader, query_loader, model, criterion, optimizer, scheduler, args.topk)

        torch.save(model.state_dict(), os.path.join(args.train_root, 
            'checkpoint_epoch{:02d}.pth'.format(epoch_idx)))



