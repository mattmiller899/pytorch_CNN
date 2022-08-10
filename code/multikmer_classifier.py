import argparse
import random
import sys
import torch
import torch.nn as nn
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
from copy import deepcopy
import gc

"""
Run on local with:
python code/deep_learning/multikmer_classifier.py -ct -cv -i data/multikmer_test/in_small -p data/multikmer_test/pos_in -s data/multikmer_test/seq_in -a data/multikmer_test/aa_in -b 5 -e 1 -c data/multikmer_test/cont_in -ua -us -up -uc -o data/multikmer_test/test_results/test_out.txt -f data/plotting -g -sc 3 -kf 4


"""

def main(args):
    print("starting")
    torch.manual_seed(17)   # Randomly seed PyTorch
    #KMER_SIZE = args.kmer_size
    KMER_SIZES = args.kmer_sizes
    BATCH_SIZE = args.batch_size
    pos_dir = args.pos_dir
    seq_dir = args.seq_dir
    cont_dir = args.cont_dir
    aa_dir = args.aa_dir
    gv_dir = args.gv_dir
    v_dir = args.v_dir
    use_gpu = args.use_gpu
    learning_rate = args.learning_rate
    epochs = args.epochs
    check_val = args.check_val
    check_test = args.check_test
    use_seq = args.use_seq
    use_pos = args.use_pos
    use_cont = args.use_cont
    use_rev = args.use_rev
    use_aa = args.use_aa
    debug = args.debug
    out_file = args.out_file
    patience = args.patience
    fig_dir = args.fig_dir
    split_channels = args.split_channels
    num_channels = use_cont + use_pos + use_seq + use_aa if split_channels else 1
    num_convs = args.num_convs 
    num_fcs = args.num_fcs
    filter_size = args.filter_size
    kfolds = args.kfold
    histograms = args.histograms

    labels = [0, 1]
    #TODO ADD BACK N
    bases = ['A', 'T', 'G', 'C']
    max_kmer = max(KMER_SIZES)
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=max_kmer)]
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
    (tab_data, kmer_vocab, vec_sizes, num_kmers_per_read,
     train_val_fields) = utils.load_reads_create_kmers_multidir(KMER_SIZES, [gv_dir, v_dir], pos_dir, seq_dir, cont_dir, aa_dir,
                                                       use_seq, use_pos, use_cont, use_rev, use_aa)


    print(f"tab data shape = {len(tab_data.examples)}")
    print(f"tab data kmers = {tab_data.examples[0].kmers}\n tab data label = {tab_data.examples[0].labels}")
    #TODO CHANGE
    random.seed(22)
    input_vecs = kmer_vocab.vectors
    vec2idx = {tuple(vec.numpy()): idx for idx, vec in enumerate(input_vecs)}
    idx2ntide = kmer_vocab.itos
    NUM_KMERS = len(tab_data.examples[0].kmers)
    if not split_channels:
        vec_sizes = [sum(vec_sizes)]
    #split data is a tuple
    #split_data = utils.split_dataset(tab_data, kfolds)
    aurocs = []
    auprcs = []
    with open(out_file, "w") as out:
        #Create model, train, and test k-fold times
        for curr_fold, (train_exs, test_exs) in enumerate(utils.split_dataset(tab_data, kfolds)):
            print(f"training")
            train_iter, val_iter, test_iter = utils.create_iterators(train_exs, test_exs, BATCH_SIZE, curr_fold, train_val_fields)
            #Create model
            cnn = mods.KmerCNN(input_vecs, NUM_KMERS, num_channels, num_convs, num_fcs, vec_sizes, filter_size=filter_size,
                               use_gpu=use_gpu)
            cnn.apply(utils.init_weights)
            if debug:
                utils.check_memory_usage()
            if use_gpu:
                cnn = cnn.to("cuda")
            if debug:
                utils.check_memory_usage()
            """
            print(
                f"memory allocated = {torch.cuda.memory_allocated('cuda:0')}\nmem avail = "
                f"{torch.cuda.max_memory_allocated('cuda:0')}\nmem cached = {torch.cuda.memory_cached()}\nmax cached = {torch.cuda.max_memory_cached()}")
            exit()
            """
            cnn_optimizer = optim.Adam(cnn.parameters(), learning_rate)
            criterion = nn.BCELoss()
            curr_epoch = 1
            es_counter = 0
            val_loss_arr = []
            train_loss_arr = []
            lowest_loss = 10000

            with tqdm(total=epochs, desc=f"Training") as pbar:
                print("Begin Training")
                #for epoch in range(epochs):
                while es_counter < patience and curr_epoch <= epochs:
                    # Set the model to training mode
                    cnn.train()
                    total_loss = 0
                    print("before epoch mem check\n\n")
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                print(type(obj), obj.size())
                        except:
                            pass
                    for batch_count, batch in enumerate(train_iter):
                        # Clear the model gradients and current gradient
                        cnn.zero_grad()
                        cnn_optimizer.zero_grad()
                        kmer_data = batch.kmers
                        truth = batch.labels.squeeze(1).float()
                        # Send data to GPU if available
                        if use_gpu:
                            kmer_data = kmer_data.cuda()
                            truth = truth.cuda()
                        if debug:
                            print(f"kmer data size = {kmer_data.size()}")
                        pred = cnn(kmer_data).view(truth.size(0))
                        #print(f"pred = {pred}\ntruth = {truth}")
                        loss = criterion(pred, truth)
                        #print(
                        #    f"memory allocated = {torch.cuda.memory_allocated('cuda:0')}\nmem avail = {torch.cuda.max_memory_allocated('cuda:0')}")
                        # Propagate loss
                        if debug:
                            utils.check_memory_usage()
                        loss.backward()
                        # Update the optimizer
                        cnn_optimizer.step()
                        new_loss = loss.item()
                        total_loss += float(new_loss)
                        """
                        print(f"after batch {batch_count} mem check\n\n")
                        for obj in gc.get_objects():
                            try:
                                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                    print(type(obj), obj.size())
                            except:
                                pass
                        """
                    """
                    print("after epoch mem check\n\n")
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                print(type(obj), obj.size())
                        except:
                            pass
                    """
                    train_loss = total_loss / len(train_iter)
                    train_loss_arr.append(train_loss)
                    pbar.set_postfix(loss=train_loss)
                    pbar.update()
                    curr_epoch += 1
                    #print(f"Epoch memory_alloc = {torch.cuda.memory_allocated()}\nmemory_cached = {torch.cuda.memory_cached()}")

                    #Check val
                    if check_val:
                        cnn.eval()
                        total_val_loss = 0
                        with torch.no_grad():
                            for batch in val_iter:
                                kmer_data = batch.kmers
                                truth = batch.labels.float()
                                # Send data to GPU if available
                                if use_gpu:
                                    kmer_data = kmer_data.cuda()
                                    truth = truth.cuda()

                                pred = cnn(kmer_data).view(truth.size(0))
                                total_val_loss += float(criterion(pred, truth).item())

                            val_loss = total_val_loss / len(val_iter)
                            val_loss_arr.append(val_loss)
                            print(f"\nEpoch {curr_epoch-1} val loss = {total_val_loss}")
                            if lowest_loss > val_loss:
                                lowest_loss = val_loss
                                es_counter = 0
                            else:
                                es_counter += 1

            if es_counter == patience:
                print("EARLY STOPPING TRIGGERED")


            #print(f"Test memory_alloc = {torch.cuda.memory_allocated()}\nmemory_cached = {torch.cuda.memory_cached()}")

            if check_test:
                #TODO CHANGE TO MIN
                if histograms:
                    pred1_kmers_dict = {''.join(p):0 for p in itertools.product(bases, repeat=max_kmer)}
                    pred2_kmers_dict = deepcopy(pred1_kmers_dict)
                    true1_kmers_dict = deepcopy(pred1_kmers_dict)
                    true2_kmers_dict = deepcopy(pred1_kmers_dict)

                    pred_kmers_dict = [pred1_kmers_dict, pred2_kmers_dict]
                    true_kmers_dict = [true1_kmers_dict, true2_kmers_dict]
                    pred_kmer_counts, true_kmer_counts = [0, 0], [0, 0]
                cnn.eval()
                with torch.no_grad():
                    preds = []
                    truths = []
                    round_preds = []
                    total_test_loss = 0
                    for batch in test_iter:
                        kmer_data = batch.kmers
                        truth = batch.labels.float()
                        # Send data to GPU if available
                        if use_gpu:
                            kmer_data = kmer_data.cuda()
                            truth = truth.cuda()
                        #print(f"Testbatch memory_alloc = {torch.cuda.memory_allocated()}\nmemory_cached = {torch.cuda.memory_cached()}")

                        pred = cnn(kmer_data).view(truth.size(0))
                        total_test_loss += float(criterion(pred, truth))
                        round_pred = pred.round().cpu().numpy()
                        round_preds.extend(round_pred)
                        preds.extend(pred.cpu().numpy())
                        truths.extend(truth.cpu().numpy())
                        if histograms:
                            for i, tmp_pred in enumerate(round_pred):
                                tmp_pred = int(tmp_pred)
                                tmp_true = int(truth[i])
                                for j in range(kmer_data.size(0)):
                                    tmp_ntide = idx2ntide[kmer_data[j, i]]
                                    if len(tmp_ntide) < max_kmer or "<" in tmp_ntide:
                                    #if len(tmp_ntide) > min(KMER_SIZES) or "<" in tmp_ntide:
                                        continue
                                    pred_kmers_dict[tmp_pred][tmp_ntide] += 1
                                    true_kmers_dict[tmp_true][tmp_ntide] += 1
                                    #kmer_count_list[tmp_pred].append(tmp_ntide)
                                pred_kmer_counts[tmp_pred] += num_kmers_per_read
                                true_kmer_counts[tmp_true] += num_kmers_per_read

                    acc = accuracy_score(truths, round_preds)
                    f1 = f1_score(truths, round_preds)
                    auprc = average_precision_score(truths, preds)
                    auprcs.append(auprc)
                    auroc = roc_auc_score(truths, preds)
                    aurocs.append(auroc)
                    prec = precision_score(truths, round_preds)
                    rec = recall_score(truths, round_preds)
                    test_loss = total_test_loss / len(test_iter)
                    out.write(f"Fold {curr_fold}\nNum_epochs ran = {curr_epoch}\ntest loss = {test_loss:.4f}\n"
                              f"acc = {acc:.4f}\nf1 = {f1:.4f}\nprec = {prec:.4f}\nrec = {rec:.4f}\n"
                              f"auprc = {auprc:.4f}\nauroc = {auroc:.4f}\n\n")


                    #Check misclassified reads

                    #Create kmer histogram
                    if histograms:
                        print(f"pred kmer counts = {pred_kmer_counts} sum = {sum(pred_kmer_counts)}\ntrue kmer_counts = {true_kmer_counts} sum = {sum(true_kmer_counts)}")
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(40, 20))
                        pred_true_max = max(max([max(pred_kmers_dict[0].values()), max(pred_kmers_dict[1].values())]), max([max(true_kmers_dict[0].values()), max(true_kmers_dict[1].values())]))
                        print(f"pred true max = {pred_true_max}")
                        pred_x1 = [(i / pred_kmer_counts[0]) / pred_true_max for i in pred_kmers_dict[0].values()]
                        pred_x2 = [(i / pred_kmer_counts[1]) / pred_true_max for i in pred_kmers_dict[1].values()]
                        true_x1 = [(i / true_kmer_counts[0]) / pred_true_max for i in true_kmers_dict[0].values()]
                        true_x2 = [(i / true_kmer_counts[1]) / pred_true_max for i in true_kmers_dict[1].values()]
                        print(f"pred x1 = {pred_x1}")
                        pred_x1_sorted, pred_x2_sorted = zip(*sorted(zip(pred_x1, pred_x2)))
                        true_x1_sorted, true_x2_sorted = zip(*sorted(zip(true_x1, true_x2)))
                        print(f"sum pred_x1 = {sum(pred_x1)}\nsum pred_x2 = {sum(pred_x2)}\nsum true_x1 = {sum(true_x1)}\nsum true_x2 = {sum(true_x2)}")
                        print(f"lens = {len(pred_x1)} {len(pred_x2)} {len(true_x1)} {len(true_x2)}")
                        #exit()
                        ax1.bar(list(range(len(pred_x1))), pred_x1, 1, color="b", alpha=0.75)
                        ax1.bar(list(range(len(pred_x2))), pred_x2, 1, color="g", alpha=0.75)
                        ax2.bar(list(range(len(true_x1))), true_x1, 1, color="b", alpha=0.75)
                        ax2.bar(list(range(len(true_x2))), true_x2, 1, color="g", alpha=0.75)
                        ax3.bar(list(range(len(pred_x1_sorted))), pred_x1_sorted, 1, color="b", alpha=0.75)
                        ax3.bar(list(range(len(pred_x2_sorted))), pred_x2_sorted, 1, color="g", alpha=0.75)
                        ax4.bar(list(range(len(true_x1_sorted))), true_x1_sorted, 1, color="b", alpha=0.75)
                        ax4.bar(list(range(len(true_x2_sorted))), true_x2_sorted, 1, color="g", alpha=0.75)
                        #ax1.grid(True)
                            #print(f"all kmers = {all_kmers}\nlen all_kmers = {len(all_kmers)}")
                        prev_nuc = "Z"
                        xlabs = []
                        changing_pos = max_kmer - (max_kmer - 1)
                        for curr_kmer in all_kmers:
                            curr_nuc = curr_kmer[max_kmer - changing_pos - 1]
                            if curr_nuc != prev_nuc:
                                prev_nuc = curr_nuc
                                xlabs.append(curr_kmer[:-1] + "X")
                            else:
                                xlabs.append("")
                        print(f"xlabs = {xlabs} len = {len(xlabs)}")
                        ax1.set_xticks(list(range(len(pred_x1))))
                        ax1.set_xticklabels(xlabs, rotation=90)
                        ax1.set_ylabel("Normalized Kmer Counts")
                        ax1.set_title("Normalized K-mer counts of predicted classes (Unsorted)")
                        ax2.set_xticks(list(range(len(pred_x1))))
                        ax2.set_xticklabels(xlabs, rotation=90)
                        ax2.set_ylabel("Normalized Kmer Counts")
                        ax2.set_title("Normalized K-mer counts of true classes (Unsorted)")
                        ax3.set_xlabel("Kmers")
                        ax3.set_ylabel("Normalized Kmer Counts")
                        ax3.set_title("Normalized K-mer counts of predicted classes (Sorted)")
                        ax4.set_xlabel("Kmers")
                        ax4.set_ylabel("Normalized Kmer Counts")
                        ax4.set_title("Normalized K-mer counts of true classes (Sorted)")
                        #plt.savefig(f"{fig_dir}/unsorted_hist.png")
                        #plt.cla()
                        #plt.show()
                        #plt.title("K-mer counts of predicted classes (Sorted)")
                        plt.savefig(f"{fig_dir}/hists{curr_fold}.png")

            #Create loss figure
            xs = list(range(1, curr_epoch))
            plt.plot(xs, train_loss_arr, "-r", label="train")
            if check_val:
                plt.plot(xs, val_loss_arr, "-b", label="validation")
            if check_test:
                plt.plot(curr_epoch, test_loss, "ok", label="test")
            plt.legend(loc='upper right')
            plt.ylabel("Loss (BCE)")
            plt.xlabel("Epochs")

            plt.title(f"Train, Validation, Test Loss for {os.path.basename(fig_dir)} Fold {curr_fold}")
            plt.savefig(f"{fig_dir}/loss_fig{curr_fold}.png")
            plt.clf()
        out.write(f"\nAUROC Average = {np.average(aurocs):.4f}\nAUROC Standard Deviation = {np.std(aurocs):.4f}\n"
                  f"AUPRC Average = {np.average(auprcs):.4f}\nAUPRC Standard Deviation = {np.std(auprcs):.4f}\n")
        print(f"Successfully completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int,
                        help="number of epochs to train",
                        default=1000)


    parser.add_argument("-kf", "--kfold", type=int,
                        help="Number of iterations of k-fold cross val to perform",
                        default=1)

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

    parser.add_argument("-ur", "--use_rev", dest="use_rev", action="store_true",
                        help="indicate whether to use reverse complement")
    parser.set_defaults(use_rev=False)

    parser.add_argument("-ua", "--use_aa", dest="use_aa", action="store_true",
                        help="indicate whether to use amino acid representations in addition to kmers (only works for 3/6/9/etc)")
    parser.set_defaults(use_aa=False)
    
    parser.add_argument("-cv", "--check_val", dest="check_val", action="store_true",
                        help="indicate whether to check validation set after every epoch")
    parser.set_defaults(check_val=False)
    
    parser.add_argument("-ct", "--check_test", dest="check_test", action="store_true",
                        help="indicate whether to check test set after every epoch")
    parser.set_defaults(check_test=False)

    parser.add_argument("-db", "--debug", dest="debug", action="store_true",
                        help="indicate whether to print debugging statements")
    parser.set_defaults(debug=True)

    parser.add_argument("-sc", "--split_channels", dest="split_channels", action="store_true",
                        help="indicate whether to use separate channels for the seq/pos/cont embeddings")
    parser.set_defaults(split_channels=False)

    parser.add_argument("-hi", "--histograms", dest="histograms", action="store_true",
                        help="indicate whether to generate kmer frequency histograms")
    parser.set_defaults(histograms=False)

    parser.add_argument("-p", "--pos_dir", type=str,
                        help="path to dir of positional embeddings")
    parser.add_argument("-c", "--cont_dir", type=str,
                        help="path to dir of contextual embeddings")
    parser.add_argument("-s", "--seq_dir", type=str,
                        help="path to dir of positional embeddings")
    parser.add_argument("-a", "--aa_dir", type=str,
                        help="path to dir of amino acid contextual embeddings")
    parser.add_argument("-gd", "--gv_dir", type=str,
                        help="path to directory of girus reads")
    parser.add_argument("-vd", "--v_dir", type=str,
                        help="path to directory of virus reads")
    parser.add_argument("-o", "--out_file", type=str,
                        help="path to output file")
    parser.add_argument("-f", "--fig_dir", type=str,
                        help="path where plots will be saved")

    parser.add_argument('kmer_sizes', metavar='kmer_sizes', type=int, nargs='+',
                        help='kmer sizes to use')

    args = parser.parse_args()
    main(args)

