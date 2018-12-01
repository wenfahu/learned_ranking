import deepdish as dd
import torch
import argparse
import torch.nn as nn
from torch.utils.data import TensorDataset
from resnet import resnet18
from batch_sampler import ClassBalancedSampler
from torch.autograd import Variable
import os


def train(gallery_loader, query_loader, model, criterion, optimizer, scheduler, topk):
    """TODO: Docstring for main.

    :arg1: TODO
    :returns: TODO

    """
    num_episodes = len(query_loader)
    for idx in range(num_episodes):
        gallery_f, gallery_l = gallery_loader.__iter__().next()
        query_f, query_l = query_loader.__iter__().next()

        dist = torch.mm(query_f, gallery_f.transpose(1, 0))
        sorted_dist, sorted_idx = torch.sort(dist, dim=1)

        k_nearest_gallery = gallery_f.repeat(query_f.size(0), 1)[sorted_idx[:, :topk]]
        # k_nearest_gallery of shape QueryBatch x K x FeatDim
        k_nearest_gallery_label = torch.gather(gallery_l.repeat(query_f.size(0), 1), 
                1, sorted_idx)[:, :topk]

        query_gallery = torch.cat((query_f.unsqueeze(1), k_nearest_gallery), dim=1)
        # query_gallery of shape QueryBatch x (K +1 ) x FeatDim
        relations = (k_nearest_gallery.unsqueeze(2) - query_gallery.unsqueeze(1)) ** 2
        # relations of shape QueryBatch x K x (K+1) x FeatDim
        relations = relations.permute(0, 3, 1, 2)

        relations_label = query_l.unsqueeze(1) == k_nearest_gallery_label
        # relation label of shape QueryBatch x K

        scheduler.step()

        relations = relations.cuda()
        relations_label = relations_label.cuda()

        optimizer.zero_grad()
        out = model(relations)
        loss = criterion(out, relations_label.float())

        loss.backward()

        optimizer.step()

        print("\tTraining Episode {}, loss {:.3f}".format(idx, loss.item()))
        if idx % 50 == 49:
            torch.save(model.state_dict(), 'ranks/episode_{}.pth'.format(idx) )


def batchify2(iterable, n):
    """TODO: Docstring for batchify2.
    :returns: TODO

    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx+n, l)]


def batchify(iterable, n):
    """utility function for load iterable in batch, drop last
    :returns: TODO

    """
    return zip(*[iterable[i::n] for i in range(n)])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-path')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-root', default='./ranks')
    args = parser.parse_args()

    feats = dd.io.load(args.feat_path)

    k_gallery_feat = torch.FloatTensor(feats['k_nearest_gallery'])
    k_gallery_label = torch.LongTensor(feats['k_nearest_gallery_label'])

    query_feat = torch.FloatTensor(feats['query'])
    query_label = torch.LongTensor(feats['query_label'])


    model = resnet18(pretrained=False)
    model = model.cuda()
    criterion = nn.BCELoss()
    # criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.lr_decay)


    if not os.path.isdir(args.train_root):
        os.makedirs(args.train_root)

    for epoch_idx in range(args.num_epoch):

        print("Training Epoch {}".format(epoch_idx))
        idx = 0
        for knn_gallary_f, query_f, knn_gallery_l, query_l in zip(
                *[batchify2(p, args.batch_size) for p in
                    [k_gallery_feat, query_feat, k_gallery_label, query_label]]):

            query_gallery = torch.cat((query_f.unsqueeze(1), knn_gallary_f), dim=1)
            # query_gallery of shape QueryBatch x (K +1 ) x FeatDim
            relations = (knn_gallary_f.unsqueeze(2) - query_gallery.unsqueeze(1)) ** 2
            # relations of shape QueryBatch x K x (K+1) x FeatDim
            relations = relations.permute(0, 3, 1, 2)

            relations_label = query_l.unsqueeze(1) == knn_gallery_l


            scheduler.step()

            relations = relations.cuda()
            relations_label = relations_label.cuda()

            optimizer.zero_grad()
            out = model(relations)
            loss = criterion(out, relations_label.float())

            loss.backward()

            optimizer.step()

            print("\tTraining Episode {}, loss {:.3f}".format(idx, loss.item()))
            idx = idx + 1


        torch.save(model.state_dict(), os.path.join(args.train_root, 
            'checkpoint_epoch{:02d}.pth'.format(epoch_idx)))



