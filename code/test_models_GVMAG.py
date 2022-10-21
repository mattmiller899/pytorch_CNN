import os
import torch
import torch.optim as optim
import torch.nn as nn
import sys
import argparse
import utils as utils
import models as mods
import random
from sklearn.metrics import *
import numpy as np

def main(args):
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
    all_embs, vec_sizes = utils.load_embeddings_no_torchtext(args.kmer_sizes, args.use_cont, args.use_pos, args.use_seq,
                                                             args.use_aa, args.cont_dir, args.pos_dir, args.seq_dir,
                                                             args.aa_dir)
    # Load data
    all_reads, all_labels, kmer_dict, num_kmers_per_read = utils.read_fastas_from_dirs_CNN(
        [args.input_dir],
        args.read_size,
        args.kmer_sizes,
        args.use_stepk,
        use_rev=args.use_rev
    )
    aurocs, auprcs, f1s, recs, precs, accs, test_losses = [],[],[],[],[],[],[]
    output_file = f"{args.model_dir}/../GVMAG_performance.csv"
    with open(output_file, "w") as out:
        out.write("Accuracy\n")
        test_dataset = utils.ReadsDatasetCNN(all_reads, all_labels)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=0,
                                                      drop_last=False)
        model = mods.KmerCNN(all_embs,
                             num_kmers_per_read,
                             num_channels,
                             args.num_convs,
                             args.num_fcs,
                             vec_sizes,
                             filter_size=args.filter_size,
                             use_gpu=args.use_gpu,
                             debug=args.debug,
                             kmer_sizes=args.kmer_sizes
                             ).to(device)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        preds = []
        truths = []
        round_preds = []
        for batch_count, (data, labels) in enumerate(test_dataloader):
            data, labels = data.to(device), labels.float().to(device)
            if args.debug:
                print(f"data = {data}\ndata.shape = {data.shape}\nlabels = {labels}\nlabels shape = {labels.shape}")
            pred = model(data).view(labels.size(0))
            if args.debug:
                print(f"pred = {pred}\npred shape = {pred.shape}")
            round_pred = torch.round(pred).cpu().numpy()
            # print(f"round_pred = {round_pred}")
            round_preds.extend(round_pred)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())
        truths = np.array(truths)
        print(f"truths shape = {truths.shape}")
        print(f"unique truths = {np.unique(truths)}\nunique round_preds = {np.unique(round_preds)}")
        acc = accuracy_score(truths, round_preds)
        out.write(f"{acc}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True,
                        help="path to the GVMAG directory")
    parser.add_argumennt("-m", "--model_path", required=True,
                         help="path to the saved model")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="path to output dir")
    parser.add_argument("-cd", "--cont_dir", required=False,
                        help="path to the directory containing contextual kmer embeddings")
    parser.add_argument("-pd", "--pos_dir", required=False,
                        help="path to the dir containing positional kmer embeddings")
    parser.add_argument("-sd", "--seq_dir", required=False,
                        help="path to the dir containing sequential kmer embeddings")
    parser.add_argument("-ad", "--aa_dir", required=False,
                        help="path to the dir containing aa kmer embeddings")
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
