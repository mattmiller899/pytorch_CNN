import argparse
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import utils as utils
import models as mods
from sklearn.metrics import *
import matplotlib.pyplot as plt
import os
import glob
import itertools
import numpy as np
import time

"""
Run on local with:
python code/deep_learning/multiclass_classifier.py -ct -cv -i data/multiclass_test/in_small -p data/multikmer_test/pos_in -s data/multikmer_test/seq_in -b 5 -e 1 -c data/multikmer_test/cont_in -us -up -uc -o data/multiclass_test/results/test_out_old.txt -f data/plotting -g -sc 3
"""


def main(args):
    print("starting")
    torch.manual_seed(17)  # Randomly seed PyTorch
    # KMER_SIZE = args.kmer_size
    KMER_SIZES = args.kmer_sizes
    BATCH_SIZE = args.batch_size
    pos_dir = args.pos_dir
    seq_dir = args.seq_dir
    cont_dir = args.cont_dir
    in_dir = args.in_dir
    use_gpu = args.use_gpu
    learning_rate = args.learning_rate
    epochs = args.epochs
    check_val = args.check_val
    check_test = args.check_test
    use_seq = args.use_seq
    use_pos = args.use_pos
    use_cont = args.use_cont
    out_file = args.out_file
    patience = args.patience
    fig_dir = args.fig_dir
    split_channels = args.split_channels
    num_channels = use_cont + use_pos + use_seq if split_channels else 1
    num_convs = args.num_convs
    num_fcs = args.num_fcs
    filter_size = args.filter_size
    labels = [0, 1]
    #TODO CHANGE THIS LATER
    CURR_CLASS = args.taxa_lvl
    """
    KMER_SIZE = 3
    BATCH_SIZE = 100
    pos_fp = "../../data/BP_pos_3mer_one_hot_3hd_2000e_5batch_vectors.txt"
    seq_fp = "../../data/BP_seq_3mer_one_hot_5hd_1000e_5batch.txt"
    cont_fp = "../../data/3k_5w_100s.txt"
    in_fp = "../../data/ecoli-saprophyticus-reads.txt"
    use_gpu = True
    learning_rate = 0.001
    epochs = 1
    check_val = True
    check_test = True
    """
    taxa = ["kingdom", "phylum", "clas", "order", "family", "genus"]
    dataload_start_time = time.time()
    (train_iter, val_iter, test_iter, kmer_vocab, vec_sizes, num_classes_per_taxa) = utils.load_reads_create_kmers_multiclass(
                                                                                             KMER_SIZES, BATCH_SIZE,
                                                                                             in_dir,
                                                                                             labels, pos_dir, seq_dir,
                                                                                             cont_dir,
                                                                                             use_seq, use_pos, use_cont)
    dataload_end_time = time.time()
    print(f"Time to load data: {dataload_end_time - dataload_start_time}s")
    input_vecs = kmer_vocab.vectors
    vec2idx = {tuple(vec.numpy()): idx for idx, vec in enumerate(input_vecs)}
    idx2ntide = kmer_vocab.itos
    NUM_KMERS = len(train_iter.dataset.examples[0].kmers)
    if not split_channels:
        vec_sizes = [sum(vec_sizes)]
    curr_num_classes = num_classes_per_taxa[CURR_CLASS]
    curr_taxa = taxa[CURR_CLASS]
    print(f"curr classes = {curr_num_classes}\ncurr_taxa = {curr_taxa}")
    # Create model
    cnn = mods.KmerMulticlassCNN(input_vecs, NUM_KMERS, num_channels, num_convs, num_fcs, vec_sizes,
                                 num_classes_per_taxa[CURR_CLASS], filter_size=filter_size, use_gpu=use_gpu)

    if use_gpu:
        cnn = cnn.to("cuda")

    cnn_optimizer = optim.Adam(cnn.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    curr_epoch = 1
    es_counter = 0
    val_loss_arr = []
    train_loss_arr = []
    lowest_loss = 100

    with tqdm(total=epochs, desc=f"Training") as pbar:
        print("Begin Training")
        # for epoch in range(epochs):
        while es_counter < patience and curr_epoch <= epochs:
            # Set the model to training mode
            cnn.train()
            total_loss = 0
            for batch in train_iter:
                # Clear the model gradients and current gradient
                cnn.zero_grad()
                cnn_optimizer.zero_grad()
                kmer_data = batch.kmers
                #TODO GET CURR TAXA TRUTH
                truth = utils.get_truth(batch, curr_taxa)
                # Send data to GPU if available
                if use_gpu:
                    kmer_data = kmer_data.cuda()
                    truth = truth.cuda()
                #pred = cnn(kmer_data).view(truth.size(0))
                #print(f"truth = {truth}\ntruth size = {truth.size()}")
                pred = cnn(kmer_data).view(kmer_data.size(1), -1)
                #print(f"pred = {pred}\npred size = {pred.size()}")
                loss = criterion(pred, truth)
                # Propagate loss
                loss.backward()
                # Update the optimizer
                cnn_optimizer.step()
                new_loss = loss.item()
                total_loss += new_loss

            train_loss = total_loss / len(train_iter)
            train_loss_arr.append(train_loss)
            pbar.set_postfix(loss=train_loss)
            pbar.update()
            curr_epoch += 1
            # print(f"Epoch memory_alloc = {torch.cuda.memory_allocated()}\nmemory_cached = {torch.cuda.memory_cached()}")

            # Check val
            if check_val:
                cnn.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch in val_iter:
                        kmer_data = batch.kmers
                        truth = utils.get_truth(batch, curr_taxa)
                        # Send data to GPU if available
                        if use_gpu:
                            kmer_data = kmer_data.cuda()
                            truth = truth.cuda()

                        pred = cnn(kmer_data).view(kmer_data.size(1), -1)
                        total_val_loss += criterion(pred, truth).item()

                    val_loss = total_val_loss / len(val_iter)
                    val_loss_arr.append(val_loss)
                    print(f"\nEpoch {curr_epoch - 1} val loss = {total_val_loss}")
                    if lowest_loss > val_loss:
                        lowest_loss = val_loss
                        es_counter = 0
                    else:
                        es_counter += 1

    if es_counter == patience:
        print("EARLY STOPPING TRIGGERED")

    # print(f"Test memory_alloc = {torch.cuda.memory_allocated()}\nmemory_cached = {torch.cuda.memory_cached()}")

    if check_test:
        bases = ['A', 'T', 'G', 'C', 'N']
        # TODO CHANGE TO MIN
        class1_kmers_dict = {''.join(p): 0 for p in itertools.product(bases, repeat=max(KMER_SIZES))}
        class2_kmers_dict = {''.join(p): 0 for p in itertools.product(bases, repeat=max(KMER_SIZES))}
        pred_kmers_dict = [class1_kmers_dict, class2_kmers_dict]
        cnn.eval()
        with torch.no_grad():
            preds = []
            truths = []
            total_test_loss = 0
            for batch in test_iter:
                kmer_data = batch.kmers
                truth = utils.get_truth(batch, curr_taxa)
                # Send data to GPU if available
                if use_gpu:
                    kmer_data = kmer_data.cuda()
                    truth = truth.cuda()
                # print(f"Testbatch memory_alloc = {torch.cuda.memory_allocated()}\nmemory_cached = {torch.cuda.memory_cached()}")

                pred = cnn(kmer_data).view(kmer_data.size(1), -1)
                log_pred = F.log_softmax(pred).exp()
                _, max_pred = torch.max(log_pred, 1)
                #print(f"log pored = {log_pred}\nmax pred = {max_pred}\ntruth = {truth}")
                total_test_loss += criterion(pred, truth)
                #TODO get best prediction for each read
                preds.extend(max_pred.cpu().numpy())
                truths.extend(truth.cpu().numpy())
                #TODO figure out how to make the histograms for multiclass
                """
                for i, tmp_pred in enumerate(round_pred):
                    tmp_pred = int(tmp_pred)
                    for j in range(kmer_data.size(0)):
                        tmp_ntide = idx2ntide[kmer_data[j, i]]
                        if len(tmp_ntide) < max(KMER_SIZES) or "<" in tmp_ntide:
                            # if len(tmp_ntide) > min(KMER_SIZES) or "<" in tmp_ntide:
                            continue
                        pred_kmers_dict[tmp_pred][tmp_ntide] += 1
                        # kmer_count_list[tmp_pred].append(tmp_ntide)
                """

            acc = accuracy_score(truths, preds)
            f1_mac = f1_score(truths, preds, average="macro")
            f1_wei = f1_score(truths, preds, average="weighted")
            prec_mac = precision_score(truths, preds, average="macro")
            prec_wei = precision_score(truths, preds, average="weighted")
            rec_mac = recall_score(truths, preds, average="macro")
            rec_wei = recall_score(truths, preds, average="weighted")
            test_loss = total_test_loss / len(test_iter)
            if use_gpu:
                test_loss = test_loss.cpu().detach().numpy()
            else:
                test_loss = test_loss.detach().numpy()
            with open(out_file, "w") as out:
                out.write(
                    f"Num_epochs ran = {curr_epoch}\ntest loss = {test_loss:.4f}\nacc = {acc:.4f}\nf1_mac = {f1_mac:.4f}"
                    f"\nf1_weighted = {f1_wei:.4f}\nprec_mac = {prec_mac:.4f}\nprec_weighted = {prec_wei:.4f}\n"
                    f"rec_mac = {rec_mac:.4f}\nrec_weighted = {rec_wei:.4f}")
            # Create kmer histogram
            #TODO histograms for multiclass
            """
            x1 = pred_kmers_dict[0].values()
            x2 = pred_kmers_dict[1].values()
            plt.bar(list(range(len(x1))), x1, 1, color="b", alpha=0.75)
            plt.grid(True)
            plt.ylabel("Count")
            plt.xlabel("Kmers")
            plt.title("K-mer counts of predicted classes (Unsorted)")
            plt.bar(list(range(len(x2))), x2, 1, color="g", alpha=0.75)
            plt.grid(True)
            plt.ylabel("Count")
            plt.xlabel("Kmers")
            plt.savefig(f"{fig_dir}/unsorted_hist.png")
            plt.cla()
            # Sorted histograms
            x1, x2 = zip(*sorted(zip(x1, x2)))
            plt.bar(list(range(len(x1))), x1, 1, color="b", alpha=0.75)
            plt.grid(True)
            plt.ylabel("Count")
            plt.xlabel("Kmers")
            # plt.title("Predicted Class 1 Kmer Histogram (Sorted)")
            plt.bar(list(range(len(x2))), x2, 1, color="g", alpha=0.75)
            plt.grid(True)
            plt.ylabel("Count")
            plt.xlabel("Kmers")
            plt.title("K-mer counts of predicted classes (Sorted)")
            plt.savefig(f"{fig_dir}/sorted_hist.png")
            """

    # Create loss figure
    xs = list(range(1, curr_epoch))
    plt.plot(xs, train_loss_arr, "-r", label="train")
    if check_val:
        plt.plot(xs, val_loss_arr, "-b", label="validation")
    if check_test:
        plt.plot(curr_epoch, test_loss, "ok", label="test")
    plt.legend(loc='upper right')
    plt.ylabel("Loss (BCE)")
    plt.xlabel("Epochs")

    plt.title(f"Train, Validation, Test Loss for {os.path.basename(fig_dir)}")
    plt.savefig(f"{fig_dir}/loss_fig.png")
    print(f"Successfully completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int,
                        help="number of epochs to train",
                        default=1000)

    """
    parser.add_argument("-k", "--kmer-size", type=int,
                        help="Size of kmers to generate",
                        required=True)
    """
    parser.add_argument("-r", "--learning-rate", type=float,
                        help="learning rate",
                        default=1e-3)

    parser.add_argument("-b", "--batch-size", type=int,
                        help="batch size",
                        default=100)

    parser.add_argument("-pa", "--patience", type=int,
                        help="patience, how many epochs of decreasing validation performance are allowed before early stopping",
                        default=20)

    parser.add_argument("-nc", "--num_convs", type=int,
                        help="# of convolution/pooling layers",
                        default=2)

    parser.add_argument("-nf", "--num_fcs", type=int,
                        help="# of fully connected layers",
                        default=4)

    parser.add_argument("-fs", "--filter_size", type=int,
                        help="filter size",
                        default=3)
    parser.add_argument("-tl", "--taxa_lvl", type=int,
                        help="taxa lvl to test. 0 (kingdom) through 5 (genus)",
                        default=0)

    parser.add_argument("-g", "--use_gpu", dest="use_gpu", action="store_true",
                        help="indicate whether to use CPU or GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument("-up", "--use_pos", dest="use_pos", action="store_true",
                        help="indicate whether to use positional embeddings")
    parser.set_defaults(use_pos=False)

    parser.add_argument("-us", "--use_seq", dest="use_seq", action="store_true",
                        help="indicate whether to use sequential embeddings")
    parser.set_defaults(use_seq=False)

    parser.add_argument("-uc", "--use_cont", dest="use_cont", action="store_true",
                        help="indicate whether to use contextual embeddings")
    parser.set_defaults(use_cont=False)

    parser.add_argument("-cv", "--check_val", dest="check_val", action="store_true",
                        help="indicate whether to check validation set after every epoch")
    parser.set_defaults(check_val=False)

    parser.add_argument("-ct", "--check_test", dest="check_test", action="store_true",
                        help="indicate whether to check test set after every epoch")
    parser.set_defaults(check_test=False)

    parser.add_argument("-sc", "--split_channels", dest="split_channels", action="store_true",
                        help="indicate whether to use separate channels for the seq/pos/cont embeddings")
    parser.set_defaults(split_channels=False)

    parser.add_argument("-p", "--pos_dir", type=str,
                        help="path to dir of positional embeddings")
    parser.add_argument("-c", "--cont_dir", type=str,
                        help="path to dir of contextual embeddings")
    parser.add_argument("-s", "--seq_dir", type=str,
                        help="path to dir of positional embeddings")
    parser.add_argument("-i", "--in_dir", type=str,
                        help="path to directory of reads")
    parser.add_argument("-o", "--out_file", type=str,
                        help="path to output file")
    parser.add_argument("-f", "--fig_dir", type=str,
                        help="path where plots will be saved")
    parser.add_argument('kmer_sizes', metavar='kmer_sizes', type=int, nargs='+',
                        help='kmer sizes to use')

    args = parser.parse_args()
    main(args)

