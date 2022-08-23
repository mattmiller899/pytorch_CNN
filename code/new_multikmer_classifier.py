import torch
import torch.optim as optim
import torch.nn as nn
import utils as utils
import models as mods
import argparse
import os
import sys
import math
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *
from torchsummary import summary
from pytorch_lamb import Lamb, log_lamb_rs
import random
from torch.utils.tensorboard import SummaryWriter


"""
python code/new_multikmer_classifier.py -gd data/girus -vd data/virus -ef embeddings/cont/3k_5w_100s.txt -o results/ -g -b 10 -e 1 3
"""


def main(args):
    # Load data
    random.seed(17)
    torch.manual_seed(17)  # Randomly seed PyTorch
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    print(f"device = {device}")
    # TODO change num_classes
    num_classes = 2
    if args.debug:
        print(f"num_classes = {num_classes}")
    NUM_KMERS = len(args.kmer_sizes)
    num_channels = args.use_cont + args.use_pos + args.use_seq + args.use_aa if args.split_channels else 1

    # Load embeddings
    embed_vocab = utils.load_embeddings(args.embed_file)
    print(f"embed_vocab = {embed_vocab.__dict__}")
    kmer_dict = embed_vocab.stoi
    print(f"kmer_dict = {kmer_dict}")
    # Load data
    all_reads, all_labels, kmer_dict, num_kmers_per_read = utils.read_fastas_from_dirs_CNN(
        [args.girus_dir, args.virus_dir],
        args.read_size,
        args.kmer_sizes,
        args.use_stepk,
        use_rev=args.use_rev,
        kmer_dict=kmer_dict
    )
    #print(f"all_reads = {all_reads}\nfirst read = {all_reads[0]}\nfirst read len = {len(all_reads[0])}\nnum_kmers_per_read = {num_kmers_per_read}")
    if args.debug:
        print(f"all_labels = {all_labels}\nunique labels = {np.unique(all_labels)}\nfirst read = {all_reads[0]}"
              f"\nnum_kmers_per_read = {num_kmers_per_read}")
    # TODO add support for multiple embeddings
    vec_sizes = [embed_vocab.vectors.size(1)]
    dummy_arr = np.zeros((len(all_reads)))
    dev_kf = KFold(10, shuffle=True)
    for (tmp_train_idx, dev_idx) in dev_kf.split(dummy_arr):
        break
    dev_dataset = utils.ReadsDatasetCNN(all_reads[dev_idx], all_labels[dev_idx])
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    train_test_reads, train_test_labels = all_reads[tmp_train_idx], all_labels[tmp_train_idx]
    kf = KFold(args.kfolds, shuffle=True)
    dummy_arr = np.zeros((len(train_test_reads)))
    aurocs, auprcs, f1s, recs, precs, accs, test_losses = [],[],[],[],[],[],[]
    best_dev_loss = 1000000
    best_dev_epoch = -1
    # TODO change back
    output_file = f"{args.output_dir}/r{args.read_size}_k{args.kmer_sizes}.csv"
    model_dir = f"{args.output_dir}/models"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(f"{args.output_dir}/runs"):
        os.makedirs(f"{args.output_dir}/runs")
    with open(output_file, "w") as out:
        #out.write("Fold,Epoch,Test Loss,Accuracy,F1-Score,Precision,Recall,AUPRC,AUROC\n")
        out.write("Fold,Epoch,Test Loss,Accuracy,F1-Score,Precision,Recall\n")
        loss_writer = SummaryWriter(f"{args.output_dir}/runs")
        #Create model, train, and test k-fold times
        loss_writer = SummaryWriter(f"{args.output_dir}/runs")
        for curr_fold, (train_idx, test_idx) in enumerate(kf.split(dummy_arr)):
            train_dataset = utils.ReadsDatasetCNN(train_test_reads[train_idx], train_test_labels[train_idx])
            test_dataset = utils.ReadsDatasetCNN(train_test_reads[test_idx], train_test_labels[test_idx])
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                          drop_last=False)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                          drop_last=False)
            #Create model
            model = mods.KmerCNN(embed_vocab.vectors,
                                num_kmers_per_read,
                                num_channels,
                                args.num_convs,
                                args.num_fcs,
                                vec_sizes,
                                filter_size=args.filter_size,
                                use_gpu=args.use_gpu,
                                debug=args.debug
                                 ).to(device)
            model.apply(utils.init_weights)
            opt = optim.Adam(model.parameters(), args.learning_rate)
            #TODO test other loss functions
            criterion = nn.BCELoss()
            curr_epoch = 0
            #train_loss_arr = []
            print(f"len train_dataloader = {len(train_dataloader)}")
            if args.debug:
                print(f"mem usage before training")
                utils.print_mem_usage(device)
            while curr_epoch < args.epochs:
                model.train()
                total_loss = 0.0
                running_loss = 0.0
                for batch_count, (data, labels) in enumerate(train_dataloader):
                    # Clear the model gradients and current gradient
                    data, labels = data.to(device), labels.float().to(device)
                    #writer.add_graph(model, data)
                    #writer.close()
                    model.zero_grad()
                    opt.zero_grad()
                    if args.debug:
                        print(f"data = {data}\ndata.shape = {data.shape}\nlabels = {labels}\nlabels shape = {labels.shape}")
                    pred = model(data).view(labels.size(0))
                    if args.debug:
                        print(f"pred = {pred}\npred shape = {pred.shape}")
                    loss = criterion(pred, labels)
                    #loss = criterion(torch.log(pred), labels)
                    loss.backward()
                    opt.step()
                    curr_loss = float(loss.item())
                    total_loss += curr_loss
                    running_loss += curr_loss
                    #if args.debug:
                    #    print(f"mem usage after batch {batch_count}")
                    #    utils.print_mem_usage(device)
                    #    if batch_count == 2:
                    #       exit()
                    print(f"batch {batch_count}")
                    if batch_count % args.log_interval == args.log_interval - 1:
                        loss_writer.add_scalar(f'Training Loss/fold {curr_fold}',
                                               running_loss / args.log_interval,
                                               curr_epoch * len(train_dataloader) + batch_count)
                        running_loss = 0.0
                train_loss = total_loss / len(train_dataloader)
                #train_loss_arr.append(train_loss)
                #Evaluate model with dev set
                model.eval()
                total_dev_loss = 0.0
                print(f"len dev_dataloader = {len(dev_dataloader)}")
                if args.debug:
                    print("mem usage after training")
                    utils.print_mem_usage(device)
                with torch.no_grad():
                    for data, labels in dev_dataloader:
                        data, labels = data.to(device), labels.float().to(device)
                        model.zero_grad()
                        opt.zero_grad()
                        if args.debug:
                            print(f"data = {data}\nlabels = {labels}")
                        pred = model(data).view(labels.size(0))
                        if args.debug:
                            print(f"pred = {pred}\npred shape = {pred.shape}")
                        total_dev_loss += float(criterion(pred, labels).item())
                avg_dev_loss = total_dev_loss / len(dev_dataloader)
                if avg_dev_loss < best_dev_loss:
                    best_dev_epoch = curr_epoch
                    best_dev_loss = avg_dev_loss
                print(f"Epoch {curr_epoch} train loss = {train_loss:.4f} dev loss = {avg_dev_loss:.4f}")
                model_path = f"{model_dir}/model_r{args.read_size}_k{args.kmer_sizes}_nf{args.num_fcs}_" \
                             f"fs{args.filter_size}_nc{args.num_convs}_epoch{curr_epoch}.pt"
                torch.save(model.state_dict(), model_path)
                curr_epoch += 1
            #Test model
            #First load the best model based on the dev dataset
            print(f"best epoch = {best_dev_epoch}")
            best_model_path = f"{model_dir}/model_r{args.read_size}_k{args.kmer_sizes}_nf{args.num_fcs}_" \
                             f"fs{args.filter_size}_nc{args.num_convs}_epoch{best_dev_epoch}.pt"
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            #test_loss, acc, f1, prec, rec, auprc, auroc = utils.test_perceiver(model, test_dataloader, criterion, device, args, num_classes)
            # TODO finish testing
            test_loss, acc, f1, prec, rec = utils.test_CNN(model, test_dataloader, criterion, device, args, num_classes)

            test_losses.append(test_loss)
            accs.append(acc)
            f1s.append(f1)
            precs.append(prec)
            recs.append(rec)
            #auprcs.append(auprc)
            #aurocs.append(auroc)
            #out.write(f"{curr_fold},{best_dev_epoch},{test_loss:.4f},{acc:.4f},{f1:.4f},{prec:.4f},{rec:.4f},"
            #          f"{auprc:.4f},{auroc:.4f}\n")
            out.write(f"{curr_fold},{best_dev_epoch},{test_loss:.4f},{acc:.4f},{f1:.4f},{prec:.4f},{rec:.4f}\n")
        """
        out.write(f"-1,-1,{np.average(test_losses):.4f},{np.average(accs):.4f},{np.average(f1s):.4f},"
                  f"{np.average(precs):.4f},{np.average(recs):.4f},{np.average(auprcs):.4f},{np.average(aurocs):.4f}\n")
        out.write(f"-2,-2,{np.std(test_losses):.4f},{np.std(accs):.4f},{np.std(f1s):.4f},{np.std(precs):.4f},"
                  f"{np.std(recs):.4f},{np.std(auprcs):.4f},{np.std(aurocs):.4f}\n")
        """
        out.write(f"-1,-1,{np.average(test_losses):.4f},{np.average(accs):.4f},{np.average(f1s):.4f},"
                  f"{np.average(precs):.4f},{np.average(recs):.4f}\n")
        out.write(f"-2,-2,{np.std(test_losses):.4f},{np.std(accs):.4f},{np.std(f1s):.4f},{np.std(precs):.4f},"
                  f"{np.std(recs):.4f}\n")
        print(f"Successfully completed")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gd", "--girus_dir", required=True,
                        help="path to the girus input dir")
    parser.add_argument("-vd", "--virus_dir", required=True,
                        help="path to the virus input dir")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="path to output dir")
    parser.add_argument("-ef", "--embed_file", required=True,
                        help="path to the file containing kmer embeddings")
    parser.add_argument("-kf", "--kfolds", type=int,
                        help="number of kfolds to split for training/testing",
                        default=4)
    parser.add_argument("-e", "--epochs", type=int,
                        help="number of epochs to train",
                        default=1)
    parser.add_argument("-a", "--learning-rate", type=float,
                        help="learning rate",
                        default=4e-3)
    parser.add_argument("-r", "--read_size", type=int,
                        help="read length",
                        default=100)
    parser.add_argument("-b", "--batch-size", type=int,
                        help="batch size",
                        default=100)
    parser.add_argument("-g", "--use_gpu", dest="use_gpu", action="store_true",
                        help="indicate whether to use CPU or GPU")
    parser.set_defaults(use_gpu=False)
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="print all the statements")
    parser.set_defaults(debug=False)

    parser.add_argument("-uk", "--use_stepk", dest="use_stepk", action="store_false",
                        help="split reads into non-overlapping kmers")
    parser.set_defaults(use_stepk=True)
    parser.add_argument("-li", "--log-interval", type=int,
                        help="number of training batches between each log",
                        default=10)

    parser.add_argument("-sc", "--split_channels", dest="split_channels", action="store_true",
                        help="indicate whether to use separate channels for the seq/pos/cont embeddings")
    parser.set_defaults(split_channels=False)

    parser.add_argument("-up", "--use_pos", dest="use_pos", action="store_true",
                        help="indicate whether to use positional embeddings")
    parser.set_defaults(use_pos=False)

    parser.add_argument("-us", "--use_seq", dest="use_seq", action="store_true",
                        help="indicate whether to use sequential embeddings")
    parser.set_defaults(use_seq=False)

    parser.add_argument("-uc", "--use_cont", dest="use_cont", action="store_true",
                        help="indicate whether to use contextual embeddings")
    parser.set_defaults(use_cont=False)

    parser.add_argument("-ur", "--use_rev", dest="use_rev", action="store_true",
                        help="indicate whether to use reverse complement")
    parser.set_defaults(use_rev=False)

    parser.add_argument("-ua", "--use_aa", dest="use_aa", action="store_true",
                        help="indicate whether to use amino acid representations in addition to kmers (only works for 3/6/9/etc)")
    parser.set_defaults(use_aa=False)

    parser.add_argument("-nc", "--num_convs", type=int,
                        help="# of convolution/pooling layers",
                        default=2)

    parser.add_argument("-nf", "--num_fcs", type=int,
                        help="# of fully connected layers",
                        default=4)

    parser.add_argument("-fs", "--filter_size", type=int,
                        help="filter size",
                        default=3)

    parser.add_argument('kmer_sizes', metavar='kmer_sizes', type=int, nargs='+',
                        help='kmer sizes to use')
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    main(args)
