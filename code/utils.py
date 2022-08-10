from pathlib import Path
import pickle
import csv
import os
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from torchtext import data
from torchtext import vocab
import torchtext
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.utils.data
from math import floor
import glob
from Bio import SeqIO, Seq
#import h5py
import gc
from sklearn.model_selection import KFold
import itertools
from copy import deepcopy
from sklearn.metrics import *
#import h5py
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re

#CODE_CORPUS = Path("/rsgrps/bhurwitz/taxonomic_class/data/char_corpus")

def translate(seq):
    table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
        'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
    }
    protein = "AA_"
    if len(seq)%3 == 0:
        for i in range(0, len(seq), 3):
            codon = seq[i:i + 3]
            protein += table[codon]
    return protein

"""
class dataset_h5(torchtext.data.Dataset):
    def __init__(self, in_file):
        super(dataset_h5, self).__init__()
        self.file = h5py.File(in_file, 'r')
        self.n_images, self.nx, self.ny = self.file['images'].shape

    def __getitem__(self, index):
        input = self.file['images'][index,:,:]
        return input.astype('float32')

    def __len__(self):
        return self.n_images
"""


def output_size(h, w, pad, dil, fil, strd):
    new_h = floor(((h + (2 * pad) - (dil * (fil - 1)) - 1) / strd) + 1)
    new_w = floor(((w + (2 * pad) - (dil * (fil - 1)) - 1) / strd) + 1)
    return (new_h, new_w)


def OLD_accuracy_score(data):
    """
    Given a set of (predictions, truth labels), return the accuracy of the predictions.

    :param data: [List[Tuple]] -- A list of predictions with truth labels
    :returns:    [Float] -- Accuracy metric of the prediction data
    """
    return 100 * sum([1 if p == t else 0 for p, t in data]) / len(data)


def hamming_distance(pred_seq, truth_seq):
    """
    Returns the hamming distance between a generated sequence and the ground
    truth sequence.
    """
    return sum([int(p != t) for p, t in zip(pred_seq, truth_seq)])


class KmerFreqDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.kmers = torch.tensor(data[:, 1:])
        self.labels = torch.tensor(data[:, 0])

    def __len__(self):
        return self.kmers.size(0)

    def __getitem__(self, idx):
        return (self.kmers[idx], self.labels[idx])

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.indices = []

def load_kmer_freqs(batch_size, in_file, use_gpu):
    in_data = np.genfromtxt(in_file, delimiter=",")
    print(in_data)
    kmer_dataset = KmerFreqDataset(in_data)
    dataset_size = len(kmer_dataset)
    indices = list(range(dataset_size))
    val_split, test_split = int(floor(0.85 * dataset_size)), int(floor(0.05 * dataset_size))
    train_split = dataset_size - val_split - test_split

    train_data, val_data, test_data = torch.utils.data.random_split(kmer_dataset, [train_split, val_split, test_split])

    #Remove pin memory if GPU memory issues occur and push to cuda every time instead
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                             pin_memory=use_gpu)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                           pin_memory=use_gpu)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                            pin_memory=use_gpu)
    return train_iter, val_iter, test_iter




def load_generation_data(kmer_size, batch_size, in_file, pos_file, seq_file, cont_file, use_seq, use_pos, use_cont):
    """
    This function loads all data necessary for training and evaluation of a
    code/comment generation model. Data is loaded from a TSV file that
    contains all data instances. This file is found in the directory pointed to
    by data_path. Training, dev, and testing sets are created for the model
    using a torchtext BucketIterator that creates batches, indicated by the
    batch_size variable such that the batches have minimal padding. This
    function also loads pretrained word embedding vectors that are located in
    the data_path directory.

    :param: kmer_size: [int] -- size of kmers (w/o the BoK/EoK)
            batch_size: [int] -- amount of data elements per batch
    :returns: [Tuple] -- (TRAIN set of batches,
                          DEV set of batches,
                          TEST set of batches,
                          vocab)
    """
    # Create a field variable for each field that will be in our TSV file
    features_field = data.Field(sequential=True, tokenize=lambda s: s.split(" "),
                                include_lengths=False, use_vocab=True)
    labels_field = data.Field(sequential=False, include_lengths=False, use_vocab=False)

    # Used to create a tabular dataset from TSV
    train_val_fields = [("labels", labels_field), ("kmers", features_field)]

    # Build the large tabular dataset using the defined fields
    tab_data = data.TabularDataset(in_file, "CSV", train_val_fields)

    # Load the pretrained word embedding vectors
    emb_vecs = []
    vec_sizes = []
    if use_cont:
        tmp_vec = vocab.Vectors(cont_file, os.path.dirname(cont_file))
        emb_vecs.append(tmp_vec)
        vec_sizes.append(tmp_vec.vectors.size(1))
    if use_pos:
        tmp_vec = vocab.Vectors(pos_file, os.path.dirname(pos_file))
        emb_vecs.append(tmp_vec)
        vec_sizes.append(tmp_vec.vectors.size(1))
    if use_seq:
        tmp_vec = vocab.Vectors(seq_file, os.path.dirname(seq_file))
        emb_vecs.append(tmp_vec)
        vec_sizes.append(tmp_vec.vectors.size(1))
    if len(vec_sizes) == 0:
        print("ERROR: No embedding vectors loaded. Exiting...")
        exit(1)

    # Builds the known word vocab for code and comments from the pretrained vectors
    features_field.build_vocab(tab_data, vectors=emb_vecs)
    #print(f"vec_sizes = {vec_sizes}")
    # Split the large dataset into TRAIN, DEV, TEST portions
    train_data, val_data, test_data = tab_data.split(split_ratio=[0.85, 0.05, 0.1])
    # Creates batched TRAIN, DEV, TEST sets for faster training (uses auto-batching)
    train_iter = data.BucketIterator(
        train_data,
        batch_size=batch_size,
        train=True,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )

    val_iter = data.BucketIterator(
        val_data,
        batch_size=batch_size,
        train=False,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )

    test_iter = data.BucketIterator(
        test_data,
        batch_size=batch_size,
        train=False,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )

    # We need to return the test sets and the field pretrained vectors
    return (train_iter, val_iter, test_iter, features_field.vocab, vec_sizes)

def load_all_kmers(kmer_size, include_toks):
    """
    This function loads all data necessary to extract embeddings for each kmer
    from a trained Encoder/Decoder.
    :params: kmer_size -- int specifying kmer size

    :returns: Dataset of every kmer
    """
    input_path = CODE_CORPUS / "input"
    in_field = data.Field(sequential=True, tokenize=lambda s: s.split(" "),
                            include_lengths=True, use_vocab=True)

    # Used to create a tabular dataset from TSV
    kmer_fields = [("kmers", in_field)]
    if include_toks:
        tsv_file_path = input_path / f"all_{kmer_size}mers.txt"
    else:
        tsv_file_path = input_path / f"all_{kmer_size}mers_no_toks.txt"
    tab_data = data.TabularDataset(str(tsv_file_path), "TSV", kmer_fields)
    kmers = [' '.join(k.kmers) for k in tab_data]
    return kmers



def save_translations(pairs, filepath, sep):
    """Saves a set of generated translations."""
    pairs.sort(key=lambda t: (t[1], t[2]))

    with open(filepath + "--spell-sort.txt", "w") as outfile:
        for (hamm, pred, truth) in pairs:
            outfile.write(f"{hamm}\t\t{sep.join(pred)}\t\t{sep.join(truth)}\n")

    with open(filepath + "--hamming-sort.txt", "w") as outfile:
        hamming_pairs = sorted(pairs, key=lambda t: t[0], reverse=True)
        for (hamm, pred, truth) in hamming_pairs:
            outfile.write(f"{hamm}\t\t{sep.join(pred)}\t\t{sep.join(truth)}\n")


def save_scores(s, filepath):
    """Saves a set of classifications."""
    pickle.dump(s, open(str(filepath), "wb"))


def save_embeddings(kmer_embeds, filepath):
    print(f"kmer_embeds = {kmer_embeds}")
    with open(filepath, "w") as outfile:
        for (kmer_names, embeds) in kmer_embeds:
            for (kmer_name, embed) in zip(kmer_names, embeds):
                outfile.write(f"{kmer_name},{','.join(['{:.4f}'.format(x) for x in embed])}\n")


def score_classifier(preds, truths):
    """Computes and prints the precision/recall/F1 scores from a set of predictions."""
    p = precision_score(truths, preds)
    r = recall_score(truths, preds)
    f1 = f1_score(truths, preds)
    
    return p, r, f1


def load_reads_create_kmers(kmer_sizes, in_dir, pos_dir, seq_dir, cont_dir, aa_dir,
                            use_seq, use_pos, use_cont, use_rev, use_aa):
    examples = []
    print(f"len kmer sizes = {len(kmer_sizes)}")
    features_field = data.Field(sequential=True, include_lengths=False, use_vocab=True)
    labels_field = data.Field(sequential=False, include_lengths=False, use_vocab=False)
    # Used to create a tabular dataset from fasta files
    train_val_fields = [("labels", labels_field), ("kmers", features_field)]
    examples = read_fastas_from_dir(in_dir, kmer_sizes, train_val_fields, label='multikmer',
                                    use_rev=use_rev)
    #examples = read_fastas_from_dir_multiorf(in_dir, kmer_sizes, train_val_fields, label='multikmer', use_rev=use_rev)
    tab_data = data.Dataset(examples, train_val_fields)
    print(f"ex = {tab_data.examples[0].__dict__}")
    num_kmers_per_read = len(tab_data.examples[0].kmers) - (4 * len(kmer_sizes)) # The - value is for the pairs of <s> and
                                                                                 # </s> for for/rev for each kmer size
    #Load embedding vectors
    emb_vecs, vec_sizes = load_embeds(kmer_sizes, cont_dir, pos_dir, seq_dir, aa_dir, use_cont, use_pos, use_seq, use_aa)
    # Builds the known word vocab for code and comments from the pretrained vectors
    features_field.build_vocab(tab_data, vectors=emb_vecs)
    labels_field.build_vocab(tab_data)
    print(f"vec sizes = {vec_sizes}")
    #print(f"features field vocab = {features_field.vocab.__dict__}")
    # print(f"vec_sizes = {vec_sizes}")
    # Split the large dataset into TRAIN, DEV, TEST portions
    #print(f"tab data = {tab_data.fields['kmers'].vocab.__dict__}")
    return (tab_data, features_field.vocab, vec_sizes, num_kmers_per_read, train_val_fields)


def load_reads_create_kmers_multidir(kmer_sizes, in_dirs, pos_dir, seq_dir, cont_dir, aa_dir,
                            use_seq, use_pos, use_cont, use_rev, use_aa):
    examples = []
    print(f"len kmer sizes = {len(kmer_sizes)}")
    features_field = data.Field(sequential=True, include_lengths=False, use_vocab=True)
    labels_field = data.Field(sequential=False, include_lengths=False, use_vocab=False)
    # Used to create a tabular dataset from fasta files
    train_val_fields = [("labels", labels_field), ("kmers", features_field)]
    examples = read_fastas_from_dir_multidir(in_dirs, kmer_sizes, train_val_fields, label='multikmer',
                                    use_rev=use_rev)
    #examples = read_fastas_from_dir_multiorf(in_dir, kmer_sizes, train_val_fields, label='multikmer', use_rev=use_rev)
    tab_data = data.Dataset(examples, train_val_fields)
    print(f"ex = {tab_data.examples[0].__dict__}")
    num_kmers_per_read = len(tab_data.examples[0].kmers) - (4 * len(kmer_sizes)) # The - value is for the pairs of <s> and
                                                                                 # </s> for for/rev for each kmer size
    #Load embedding vectors
    emb_vecs, vec_sizes = load_embeds(kmer_sizes, cont_dir, pos_dir, seq_dir, aa_dir, use_cont, use_pos, use_seq, use_aa)
    # Builds the known word vocab for code and comments from the pretrained vectors
    features_field.build_vocab(tab_data, vectors=emb_vecs)
    labels_field.build_vocab(tab_data)
    print(f"vec sizes = {vec_sizes}")
    #print(f"features field vocab = {features_field.vocab.__dict__}")
    # print(f"vec_sizes = {vec_sizes}")
    # Split the large dataset into TRAIN, DEV, TEST portions
    #print(f"tab data = {tab_data.fields['kmers'].vocab.__dict__}")
    return (tab_data, features_field.vocab, vec_sizes, num_kmers_per_read, train_val_fields)

def load_reads_create_kmer_freqs(kmer_sizes, in_dir, use_rev):
    print(f"len kmer sizes = {len(kmer_sizes)}")
    # Used to create a tabular dataset from fasta files
    tab_data = read_fastas_from_dir_freqs(in_dir, kmer_sizes, label='multikmer', use_rev=use_rev)
    num_kmers_per_read = len(tab_data.kmers) - (4 * len(kmer_sizes)) # The - value is for the pairs of <s> and
                                                                                 # </s> for for/rev for each kmer size
    return (tab_data, num_kmers_per_read)

def generate_kmers(k):
    return [''.join(p) for p in itertools.product(["A", "C", "G", "T"], repeat=k)]

"""
def split_dataset(tab_data, kfolds):
    split_data = [] 
    if kfolds == 4:
        tmp_split = tab_data.split(split_ratio=[0.5, 0.5])
        for tmp_dataset in tmp_split:
            split_data.extend(tmp_dataset.split(split_ratio=[0.5, 0.5]))
    else:
        print("YOU NEVER FIXED THIS, ONLY KFOLD=4 IS SUPPORTED. EXITING")
        exit(1)
    # Creates batched TRAIN, DEV, TEST sets for faster training (uses auto-batching)
    return split_data
"""

def split_dataset(tab_data, kfolds):
    kf = KFold(n_splits=kfolds, random_state=22, shuffle=True)
    try:
        examples = np.array(tab_data.examples)
    except:
        print("Tab data did not have examples. Using tab data")
        examples = tab_data
    for curr_fold, (train_index, test_index) in enumerate(kf.split(examples)):
        yield (examples[train_index], examples[test_index])

"""
def create_iterators(split_data, batch_size, curr_fold, train_val_fields):
    test_data = split_data[curr_fold]
    print(f"len split data = {len(split_data)}")
    train_val_set = [split_data[i] for i in range(len(split_data)) if i != curr_fold]
    ds_concat = train_val_set[0]
    for i in range(1, len(train_val_fields)):
        ds_concat += train_val_set[i]
    list_of_ex = [i for i in ds_concat]
    train_val_ds = data.Dataset(list_of_ex, train_val_fields)
    train_data, val_data = train_val_ds.split(split_ratio=[0.9, 0.1])
    # Creates batched TRAIN, DEV, TEST sets for faster training (uses auto-batching)
    train_iter = data.BucketIterator(
        train_data,
        batch_size=batch_size,
        train=True,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )

    val_iter = data.BucketIterator(
        val_data,
        batch_size=batch_size,
        train=False,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )
    test_iter = data.BucketIterator(
        test_data,
        batch_size=batch_size,
        train=False,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )
    return train_iter, val_iter, test_iter
"""

def create_iterators(train_exs, test_exs, batch_size, curr_fold, train_val_fields):
    train_val_ds = data.Dataset(train_exs, train_val_fields)
    train_data, val_data = train_val_ds.split(split_ratio=[0.9, 0.1])
    test_data = data.Dataset(test_exs, train_val_fields)
    # Creates batched TRAIN, DEV, TEST sets for faster training (uses auto-batching)
    train_iter = data.BucketIterator(
        train_data,
        batch_size=batch_size,
        train=True,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )

    val_iter = data.BucketIterator(
        val_data,
        batch_size=batch_size,
        train=False,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )
    test_iter = data.BucketIterator(
        test_data,
        batch_size=batch_size,
        train=False,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )
    return train_iter, val_iter, test_iter



def load_reads_create_kmers_multiclass(kmer_sizes, batch_size, in_dir, file_labels, pos_dir, seq_dir, cont_dir, use_seq, use_pos, use_cont):
    taxa = ["kingdom", "phylum", "clas", "order", "family", "genus"]
    # Used to create a tabular dataset from TSV
    label_fields = []
    for i in range(len(taxa)):
        label_fields.append(data.Field(sequential=False, include_lengths=False, use_vocab=True, unk_token=None))
    features_field = data.Field(sequential=True, include_lengths=False, use_vocab=True)
    # Used to create a tabular dataset from TSV
    train_val_fields = []
    for i in range(len(taxa)):
        train_val_fields.append((taxa[i], label_fields[i]))
    train_val_fields.append(("kmers", features_field))
    examples = read_fastas_from_dir(in_dir, kmer_sizes, train_val_fields)
    #examples = read_fastas_from_dir_multiorf(in_dir, kmer_sizes, train_val_fields)
    tab_data = data.Dataset(examples, train_val_fields)
    print(f"ex = {tab_data.examples[0].__dict__}")
    num_kmers_per_read = len(tab_data.examples[0].kmers) - (4 * len(kmer_sizes)) # The - is for the pairs of <s> and
                                                                                 # </s> for for/rev for each kmer size
    # Load embedding vectors
    emb_vecs, vec_sizes = load_embeds(kmer_sizes, cont_dir, pos_dir, seq_dir, use_cont, use_pos, use_seq)
    # Builds the known word vocab for code and comments from the pretrained vectors
    features_field.build_vocab(tab_data, vectors=emb_vecs)
    for i in range(len(taxa)):
        label_fields[i].build_vocab(tab_data)
    print(f"kingdom field = {label_fields[1].vocab.__dict__}\n num classes = {len(label_fields[0].vocab.itos) - 1}")
    num_classes_per_taxa = []
    for i in range(len(taxa)):
        num_classes_per_taxa.append(len(label_fields[i].vocab.itos))
    #print(f"features field vocab = {features_field.vocab.__dict__}")
    # print(f"vec_sizes = {vec_sizes}")
    # Split the large dataset into TRAIN, DEV, TEST portions
    random.seed(22)
    train_data, val_data, test_data = tab_data.split(split_ratio=[0.85, 0.05, 0.1])
    # Creates batched TRAIN, DEV, TEST sets for faster training (uses auto-batching)
    train_iter = data.BucketIterator(
        train_data,
        batch_size=batch_size,
        train=True,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )

    val_iter = data.BucketIterator(
        val_data,
        batch_size=batch_size,
        train=False,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )

    test_iter = data.BucketIterator(
        test_data,
        batch_size=batch_size,
        train=False,
        sort_key=lambda x: x.kmers,
        shuffle=True
    )

    # We need to return the test sets and the field pretrained vectors
    return (train_iter, val_iter, test_iter, features_field.vocab, vec_sizes, num_classes_per_taxa, num_kmers_per_read)


def load_embeds(kmer_sizes, cont_dir, pos_dir, seq_dir, aa_dir, use_cont, use_pos, use_seq, use_aa):
    emb_vecs, vec_sizes = [], []
    if use_cont:
        #TODO COMBINING VECS OF SAME TYPES/DIFF K'S
        tmp_vec = combine_embeds(kmer_sizes, cont_dir, "cont")
        emb_vecs.append(tmp_vec)
        vec_sizes.append(tmp_vec.vectors.size(1))
    if use_pos:
        #os.remove(f"{pos_dir}/*.pt")
        tmp_vec = combine_embeds(kmer_sizes, pos_dir, "pos")
        emb_vecs.append(tmp_vec)
        vec_sizes.append(tmp_vec.vectors.size(1))
    if use_seq:
        #os.remove(f"{seq_dir}/*.pt")
        tmp_vec = combine_embeds(kmer_sizes, seq_dir, "seq")
        #print(f"tmp_vec = {tmp_vec.__dict__}")
        emb_vecs.append(tmp_vec)
        vec_sizes.append(tmp_vec.vectors.size(1))
    if use_aa:
        tmp_vec = combine_embeds(kmer_sizes, aa_dir, "aa")
        # print(f"tmp_vec = {tmp_vec.__dict__}")
        emb_vecs.append(tmp_vec)
        vec_sizes.append(tmp_vec.vectors.size(1))
    if len(vec_sizes) == 0:
        print("ERROR: No embedding vectors loaded. Exiting...")
        exit(1)
    return emb_vecs, vec_sizes


def combine_embeds(kmer_sizes, in_dir, emb_type):
    rand_num = random.random()
    tmp_dir = f"./embeds"
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    tmp_emb = f"{tmp_dir}/tmp_{emb_type}{rand_num}.txt"
    with open(tmp_emb, "w") as out:
        # os.remove(f"{cont_dir}/*.pt")
        print(f"kmer sizes = {kmer_sizes}")
        for k in kmer_sizes:
            kmer_glob = glob.glob(f"{in_dir}/*{k}k*.txt")
            kmer_glob.extend(glob.glob(f"{in_dir}/*{k}mer*.txt"))
            kmer_glob.extend(glob.glob(f"{in_dir}/*{k}aa*.txt"))
            print(f"glob = {kmer_glob}")
            for in_file in kmer_glob:
                with open(in_file, "r") as f:
                    _ = f.readline()
                    for l in f:
                        out.write(f"{l.rstrip()}\n")
    return vocab.Vectors(tmp_emb, os.path.dirname(tmp_emb))

def read_fastas_from_dir_multidir(in_dirs, kmer_sizes, train_val_fields, label=None, use_rev=False, add_start_stop_toks=False,
                         step=None, label_encoder=None):
    examples = []
    for in_dir in in_dirs:
        print(f"reading files from {in_dir}")
        for count, in_file in enumerate(glob.glob(f"{in_dir}/*.fa")):
            print(f"reading {in_file}")
            for record in SeqIO.parse(in_file, "fasta"):
                if label == None and label_encoder == None:
                    labels = [int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]]
                elif label == None and label_encoder != None:
                    labels = label_encoder.transform(
                        [int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]])
                else:
                    labels = [count]
                tmp_arr = []
                #TODO MAY NEED TO CHANGE BACK LATER
                #tmp_arr.extend(labels)
                tmp_arr.append(labels)
                tmp_kmer_arr = []
                for kmer in kmer_sizes:
                    if step == None:
                        stepk = kmer
                    else:
                        stepk = step
                    seq_id = record.id
                    for_seq = str(record.seq.upper())
                    length = len(for_seq)
                    # seq and rev_comp
                    for_kmers = []
                    if add_start_stop_toks:
                        for_kmers = ["<s>"]
                    tmp_for = [for_seq[j:j + kmer] for j in range(0, length - kmer + 1, stepk)]
                    for_kmers.extend(tmp_for)
                    if add_start_stop_toks:
                        for_kmers.append("</s>")
                    tmp_kmer_arr.extend(for_kmers)

                    if use_rev:
                        rev_seq = str(Seq.reverse_complement(record.seq).upper())
                        rev_kmers = []
                        if add_start_stop_toks:
                            rev_kmers = ["<s>"]
                        tmp_rev = [rev_seq[j:j + kmer] for j in range(0, length - kmer + 1, stepk)]
                        rev_kmers.extend(tmp_rev)
                        if add_start_stop_toks:
                            rev_kmers.append("</s>")
                        tmp_kmer_arr.extend(rev_kmers)

                # print(f"tmp kmer arr = {tmp_kmer_arr}")
                tmp_arr.append(tmp_kmer_arr)
                examples.append(data.Example.fromlist(tmp_arr, train_val_fields))
                #print(f"examples = {examples[0].__dict__}")
                #exit()
    return examples


def read_fastas_from_dir_freqs(in_dir, kmer_sizes, label=None, use_rev=False):
    data = []
    kmer_counts = {}
    for i in range(len(kmer_sizes)):
        kmer = kmer_sizes[i]
        all_kmers = generate_kmers(kmer)
        for tmp_kmer in all_kmers:
            kmer_counts[tmp_kmer] = 0
    for count, in_file in enumerate(glob.glob(f"{in_dir}/*.fa")):
        print(f"reading {in_file}")
        for record in SeqIO.parse(in_file, "fasta"):
            if label == None:
                labels = [int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]]
            else:
                labels = [count]
            tmp_arr = []
            tmp_arr.extend(labels)
            tmp_kmer_counts = deepcopy(kmer_counts)
            for kmer in kmer_sizes:
                step = 1
                for_seq = str(record.seq.upper())
                length = len(for_seq)
                # seq and rev_comp
                for_kmers = [for_seq[j:j + kmer] for j in range(0, length - kmer, step)]
                for tmp_kmer in for_kmers:
                    tmp_kmer_counts[tmp_kmer] += 1
                if use_rev:
                    rev_seq = str(Seq.reverse_complement(record.seq).upper())
                    rev_kmers = [rev_seq[j:j + kmer] for j in range(0, length - kmer, step)]
                    for tmp_kmer in rev_kmers:
                        tmp_kmer_counts[tmp_kmer] += 1
            # print(f"tmp kmer arr = {tmp_kmer_arr}")
            tmp_arr.extend(list(tmp_kmer_counts.values()))
            data.append(tmp_arr)
            #print(f"examples = {examples[0].__dict__}")
            #exit()
    return KmerFreqDataset(np.array(data))


def read_fastas_from_dir_multiorf(in_dir, kmer_sizes, train_val_fields, use_step=False, label=None, use_rev=False):
    examples = []
    for count, in_file in enumerate(glob.glob(f"{in_dir}/*.fa")):
        print(f"reading {in_file}")
        for record in SeqIO.parse(in_file, "fasta"):
            if label == None:
                labels = [int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]]
            else:
                labels = [count]
            tmp_arr = []
            tmp_arr.extend(labels)
            tmp_kmer_arr = []
            for kmer in kmer_sizes:
                step = kmer
                seq_id = record.id
                for_seq = str(record.seq.upper())
                rev_seq = str(Seq.reverse_complement(record.seq).upper())
                length = len(for_seq)
                # seq and rev_comp
                for_kmers = []
                rev_kmers = []
                #TODO test without these start/end tokenst
                for i in range(kmer):
                    for_kmers.append("<s>")
                    for_kmers.extend([for_seq[j:j + kmer] for j in range(i, length - kmer, step)])
                    for_kmers.append("</s>")
                    tmp_kmer_arr.extend(for_kmers)
                    if use_rev:
                        rev_kmers.append("<s>")
                        rev_kmers.extend([rev_seq[j:j + kmer] for j in range(i, length - kmer, step)])
                        rev_kmers.append("</s>")
                        tmp_kmer_arr.extend(rev_kmers)
            # print(f"tmp kmer arr = {tmp_kmer_arr}")
            tmp_arr.append(tmp_kmer_arr)
            #print(f"len example = {len(tmp_kmer_arr)}")
            examples.append(data.Example.fromlist(tmp_arr, train_val_fields))
            #print(f"examples = {examples[0].__dict__}")
            #exit()
    return examples

def OLD_read_fastas_from_dir_gan(in_dir, label=None, use_rev=False):
    all_reads = []
    for count, in_file in enumerate(glob.glob(f"{in_dir}/*.fa")):
        print(f"reading {in_file}")
        for record in SeqIO.parse(in_file, "fasta"):
            if label == None:
                labels = [int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]]
            else:
                labels = [count]
            tmp_arr = []
            tmp_arr.extend(labels)
            tmp_arr.extend(str(record.seq.upper()))
            # seq and rev_comp
            if use_rev:
                tmp_arr.extend(str(Seq.reverse_complement(record.seq).upper()))
            # print(f"tmp kmer arr = {tmp_kmer_arr}")
            all_reads.append(tmp_arr)
            # print(f"examples = {examples[0].__dict__}")
            # exit()
    return all_reads


def get_truth(batch, taxa_name):
    truth = None
    if taxa_name == "kingdom":
        truth = batch.kingdom
    elif taxa_name == "phylum":
        truth = batch.phylum
    elif taxa_name == "clas":
        truth = batch.clas
    elif taxa_name == "order":
        truth = batch.order
    elif taxa_name == "family":
        truth = batch.family
    elif taxa_name == "genus":
        truth = batch.genus
    else:
        print(f"ERROR: invalid taxa name \"{taxa_name}\"")
        exit()
    return truth

def count_kmer_freqs(examples, kmer_sizes, bases, train_val_fields):
    all_kmers = []
    for kmer in kmer_sizes:
        all_kmers.extend([''.join(p) for p in itertools.product(bases, repeat=kmer)])
    kmer_arr = []
    X = [i.kmers for i in examples]
    Y = [i.labels for i in examples]
    for i, read in enumerate(X):
        kmer_counts = {}
        for tmp_kmer in all_kmers:
            kmer_counts[tmp_kmer] = 0
        kmer_counts["<s>"] = 0
        kmer_counts["</s>"] = 0
        for tmp_kmer in read:
            kmer_counts[tmp_kmer] += 1
        tmp_arr = [Y[i]]
        tmp_arr.append(list(kmer_counts.values()))
        kmer_arr.append(data.Example.fromlist(tmp_arr, train_val_fields))
    return data.Dataset(exs, train_val_fields)



def check_memory_usage():
    total = 0
    counter = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if len(obj.size()) > 0:
                    if obj.type() == 'torch.cuda.FloatTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 32
                    elif obj.type() == 'torch.cuda.LongTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 64
                    elif obj.type() == 'torch.cuda.IntTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 32
                    # else:
                    # Few non-cuda tensors in my case from dataloader
                print(type(obj), obj.size(), obj.type())
                counter += 1
        except Exception as e:
            pass
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.type())
                counter += 1
        except:
            pass
    """
    print("{} GB\n{} tensors".format(total / ((1024 ** 3) * 8), counter))


def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data,
                                       nonlinearity="relu")


class ReadsDataset(torch.utils.data.Dataset):
    def __init__(self, all_reads):
        self.reads = all_reads
        self.one_hots = torch.FloatTensor([[1,0,0,0],
                                         [0,1,0,0],
                                         [0,0,1,0],
                                         [0,0,0,1]])
        self.embedding = torch.nn.Embedding.from_pretrained(self.one_hots)


    def __len__(self):
        return len(self.reads)

    def __getitem__(self, idx):
        curr_read = self.reads[idx]
        sample = {"labels": torch.LongTensor(curr_read[:6]), "data": self.embedding(curr_read[6])}
        return sample

class ReadsDatasetPerceiver(torch.utils.data.Dataset):
    def __init__(self, all_reads, all_labels, kmer_embedding):
        self.num_reads = len(all_reads)
        self.labels = torch.LongTensor(all_labels)
        self.reads = torch.LongTensor(all_reads)
        self.embedding = kmer_embedding

    def __len__(self):
        return len(self.reads)

    def __getitem__(self, idx):
        curr_read = self.reads[idx]
        curr_label = self.labels[idx]
        return self.embedding(curr_read), curr_label
        #return self.embedding(torch.LongTensor(curr_read)), torch.LongTensor(curr_label)

def load_reads_gan(in_dir, read_size=100, use_rev=False, label_encoder=None):
    all_reads = read_fastas_from_dir_gan(in_dir, read_size, use_rev=use_rev, label_encoder=label_encoder)
    read_dataset = ReadsDataset(all_reads)
    return read_dataset

def read_fastas_from_dir_gan(in_dir, read_size, use_rev=True, label=None, label_encoder=None):
    all_reads = []
    nuc_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
    for count, in_file in enumerate(glob.glob(f"{in_dir}/*.fa")):
        print(f"reading {in_file}")
        for record in SeqIO.parse(in_file, "fasta"):
            if label == None and label_encoder == None:
                labels = [int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]]
            elif label == None and label_encoder != None:
                labels = label_encoder.transform([int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]])
            else:
                labels = [count]
            tmp_arr = []
            tmp_arr.extend(labels)
            tmp_nuc_arr = []
            for_seq = str(record.seq.upper())
            length = len(for_seq)
            if length < read_size:
                continue
            # seq and rev_comp
            tmp_for = []
            for j in range(0, read_size):
                try:
                    tmp_for.append(nuc_dict[for_seq[j]])
                except:
                    pass
            #tmp_for = for_seq
            tmp_nuc_arr.extend(tmp_for)
            if use_rev:
                rev_seq = str(Seq.reverse_complement(record.seq).upper())
                #tmp_rev = rev_seq
                tmp_rev = []
                for j in range(0, read_size):
                    try:
                        tmp_rev.append(nuc_dict[rev_seq[j]])
                    except:
                        pass
                tmp_nuc_arr.extend(tmp_rev)
            #TODO make tmp_nuc_arr a LongTensor before inserting
            tmp_arr.append(torch.LongTensor(tmp_nuc_arr))
            #print(f"tmp_nuc_arr = {tmp_nuc_arr}\ntmp_arr = {tmp_arr}")
            #exit()
            all_reads.append(tmp_arr)
    return all_reads

class MultiColumnLabelEncoder:
    def __init__(self, num_columns=6):
        self.num_columns = num_columns
        self.label_encoders = []
        self.classes_ = []
        for i in range(self.num_columns):
            self.label_encoders.append(LabelEncoder())


    def fit(self,X):
        for i in range(self.num_columns):
            self.label_encoders[i].fit(X[:, i])
            self.classes_.append(self.label_encoders[i].classes_)

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        for col in range(self.num_columns):
            if isinstance(output, list):
                output[col] = self.label_encoders[col].transform([output[col]])[0]
            else:
                output[:, col] = self.label_encoders[col].transform(output[:, col])
        return output

    def fit_transform(self,X):
        return self.fit(X,y).transform(X)

    def inverse_transform(self,X):
        """
        Transforms LabelEncoder IDs back into the GTDB IDs we're using
        ONLY HANDLES 1 SET OF IDs AT A TIME
        """
        inv = []
        for i in range(self.num_columns):
            inv.extend(self.label_encoders[i].inverse_transform([X[i]]))
        return inv


def load_rai_vecs(kmer_size, label_encoder, input_dir="data/gan_data/rai"):
    in_file = os.path.join(input_dir, f"binary_rai_k{kmer_size}.txt")
    rai_vec_dict = {}
    with open(in_file, "r") as f:
        rai_max, rai_min = f.readline().split(",")
        rai_min = float(rai_min)
        rai_max = float(rai_max)
        rai_data = np.loadtxt(f, delimiter=",")
        taxa_ids = label_encoder.transform(rai_data[:, 0:6])
        for i, rai_row in enumerate(rai_data[:, 6:]):
            taxa_ids_str = ",".join([str(int(j)) for j in taxa_ids[i]])
            rai_vec = [float(i) for i in rai_row]
            rai_vec_dict[taxa_ids_str] = torch.FloatTensor(rai_vec)
    return rai_vec_dict, rai_min, rai_max

def labels_to_rai_vecs(labels, rai_vecs, kmer_size):
    rai_tensor = torch.zeros((labels.size(0), 4 ** kmer_size))
    np_labels = labels.cpu().numpy()
    for i in range(labels.size(0)):
        taxa_str = ",".join([str(j) for j in np_labels[i, :]])
        rai_tensor[i, :] = rai_vecs[taxa_str]
    return rai_tensor

"""
def load_reads_create_kmers_gan(kmer_size, read_size, label_encoder,
                                in_dir, pos_dir, seq_dir, cont_dir, aa_dir,
                                use_seq, use_pos, use_cont, use_rev, use_aa):
    features_field = data.Field(sequential=True, include_lengths=False, use_vocab=True)
    labels_field = data.Field(sequential=False, include_lengths=False, use_vocab=False)
    # Used to create a tabular dataset from fasta files
    train_val_fields = [("labels", labels_field), ("kmers", features_field)]
    examples = read_fastas_from_dir(in_dir, [kmer_size], train_val_fields, use_rev=use_rev, label_encoder=label_encoder)
    tab_data = data.Dataset(examples, train_val_fields)
    print(f"ex = {tab_data.examples[0].__dict__}")
    num_kmers_per_read = len(tab_data.examples[0].kmers)

    #Load embedding vectors
    emb_vecs, vec_sizes = load_embeds([kmer_size], cont_dir, pos_dir, seq_dir, aa_dir, use_cont, use_pos, use_seq, use_aa)
    # Builds the known word vocab for code and comments from the pretrained vectors
    features_field.build_vocab(tab_data, vectors=emb_vecs)
    labels_field.build_vocab(tab_data)
    print(f"vec sizes = {vec_sizes}")
    #print(f"features field vocab = {features_field.vocab.__dict__}")
    # print(f"vec_sizes = {vec_sizes}")
    # Split the large dataset into TRAIN, DEV, TEST portions
    #print(f"tab data = {tab_data.fields['kmers'].vocab.__dict__}")
    return (tab_data, features_field.vocab, vec_sizes, num_kmers_per_read, train_val_fields)
"""

def load_reads_create_rai_vecs(kmer_size, read_size, label_encoder, in_dir, use_rev=False):
    bases = ["A", "C", "G", "T"]
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=kmer_size)]
    kmer_freqs = []
    genome_ids = []
    all_reads = []
    nuc_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
    for count, in_file in enumerate(glob.glob(f"{in_dir}/*.fa")):
        kmer_freq_dict = {a: 0 for a in all_kmers}
        for record in SeqIO.parse(in_file.rstrip(), "fasta"):
            genome_ids.append(label_encoder.transform([int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]]))
            tmp_nuc_arr = []
            for_seq = str(record.seq.upper())
            length = len(for_seq)
            if length < read_size:
                continue
            # seq and rev_comp
            tmp_for = []
            for j in range(0, read_size):
                try:
                    tmp_for.append(nuc_dict[for_seq[j]])
                except:
                    pass
            # tmp_for = for_seq
            tmp_nuc_arr.extend(tmp_for)
            if use_rev:
                rev_seq = str(Seq.reverse_complement(record.seq).upper())
                # tmp_rev = rev_seq
                tmp_rev = []
                for j in range(0, read_size):
                    try:
                        tmp_rev.append(nuc_dict[rev_seq[j]])
                    except:
                        pass
                tmp_nuc_arr.extend(tmp_rev)
            all_reads.append(torch.LongTensor(tmp_nuc_arr))
            seq = str(record.seq.upper())
            seq_len = len(seq)
            num_kmers = seq_len - kmer_size + 1
            for i in range(num_kmers):
                try:
                    kmer_freq_dict[seq[i:i + kmer_size]] += 1
                except:
                    pass
            counted_kmers = sum(kmer_freq_dict.values())
            for tmp_kmer in all_kmers:
                kmer_freq_dict[tmp_kmer] = kmer_freq_dict[tmp_kmer] / counted_kmers
            kmer_freqs.append(list(kmer_freq_dict.values()))
    kmer_freqs = np.array(kmer_freqs)
    rai, rai_min, rai_max = calculate_rai(kmer_freqs, kmer_size)
    genome_ids = np.array(genome_ids)
    # print(f"rai shape = {rai.shape}\nlabs shape = {genome_ids.shape}")
    rai_dataset = RAIDataset(rai, genome_ids, all_reads)
    return rai_dataset


def calculate_rai(kmer_freqs, kmer_size):
    kmer_1_vec = np.zeros((len(kmer_freqs), (4 ** (kmer_size - 1))), dtype=float)
    for i in range(len(kmer_freqs)):
        for j in range(4 ** (kmer_size - 1)):
            kmer_1_vec[i, j] = kmer_freqs[i, j * 4:j * 4 + 4].sum()
    kmer_1_vec = kmer_1_vec / kmer_1_vec.sum()
    rai = np.zeros((len(kmer_freqs), (4 ** kmer_size)), dtype=float)
    tmp = np.min(kmer_freqs[np.nonzero(kmer_freqs)])
    tmp_k1 = np.min(kmer_1_vec[np.nonzero(kmer_1_vec)])
    for i in range(len(kmer_freqs)):
        for j in range(4 ** (kmer_size - 1)):
            for k in range(4):
                if kmer_1_vec[i, j] == 0:
                    kmer_1_vec[i, j] = tmp_k1
                if kmer_freqs[i, j * 4 + k] == 0:
                    kmer_freqs[i, j * 4 + k] = tmp
                rai[i, j * 4 + k] = 11 * np.log(kmer_freqs[i, j * 4 + k]) + np.log(kmer_1_vec[i, j])
    rai_max = np.amax(rai)
    rai_min = np.amin(rai)
    #print(f"max = {rai_max} min = {rai_min}")
    rai = (rai - rai_min) / (rai_max - rai_min)
    return rai, rai_min, rai_max


class RAIDataset(torch.utils.data.Dataset):
    def __init__(self, rai_vecs, genome_ids, all_reads):
        self.rai_vecs = torch.FloatTensor(rai_vecs)
        self.genome_ids = torch.LongTensor(genome_ids)
        self.all_reads = all_reads
        self.one_hots = torch.FloatTensor([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
        self.embedding = torch.nn.Embedding.from_pretrained(self.one_hots)

    def __len__(self):
        return self.rai_vecs.size(0)

    def __getitem__(self, idx):
        sample = {"labels": self.genome_ids[idx], "rai": self.rai_vecs[idx], "reads": self.embedding(self.all_reads[idx])}
        return sample


def calculate_kmer_freqs(data, kmer_size):
    nuc_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
    bases = nuc_dict.keys()
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=kmer_size)]
    ret_tensor = torch.zeros((data.size(0), 4 ** kmer_size))
    for i_batch in range(data.size(0)):
        kmer_freq_dict = {a: 0.0 for a in all_kmers}
        num_kmers = data.size(2) - kmer_size + 1
        for j in range(num_kmers):
            for tmp_kmer in all_kmers:
                #if tmp_kmer != "GTA":
                #    continue
                #print(f"data = {data[i_batch, :, 0:kmer_size]} data_sum = {torch.sum(data[i_batch, :, 0:kmer_size])}\nkmer = {tmp_kmer}")
                tmp_kmer_freqs = [data[i_batch, nuc_dict[tmp_base], j+kmer_pos] for kmer_pos, tmp_base in enumerate(tmp_kmer)]
                tmp_sum = sum(tmp_kmer_freqs) / ((4 ** (kmer_size - 1) * kmer_size))
                #print(f"tmp_kmer_freqs = {tmp_kmer_freqs}\ntmp_sum = {tmp_sum}")
                kmer_freq_dict[tmp_kmer] += tmp_sum
        ret_tensor[i_batch] = torch.stack(list(kmer_freq_dict.values()))
    return ret_tensor


def calculate_rai_tensor(kmer_freqs, kmer_size, rai_min=None, rai_max=None):
    kmer_1_vec = torch.zeros((len(kmer_freqs), (4 ** (kmer_size - 1))), dtype=torch.float32)
    for i in range(len(kmer_freqs)):
        for j in range(4 ** (kmer_size - 1)):
            kmer_1_vec[i, j] = kmer_freqs[i, j * 4:j * 4 + 4].sum()
    kmer_1_vec = kmer_1_vec / kmer_1_vec.sum()
    rai = torch.zeros((len(kmer_freqs), (4 ** kmer_size)), dtype=torch.float32)
    tmp = torch.min(kmer_freqs[torch.nonzero(kmer_freqs)])
    tmp_k1 = torch.min(kmer_1_vec[torch.nonzero(kmer_1_vec)])
    for i in range(len(kmer_freqs)):
        for j in range(4 ** (kmer_size - 1)):
            for k in range(4):
                if kmer_1_vec[i, j] == 0:
                    kmer_1_vec[i, j] = tmp_k1
                if kmer_freqs[i, j * 4 + k] == 0:
                    kmer_freqs[i, j * 4 + k] = tmp
                rai[i, j * 4 + k] = 11 * torch.log(kmer_freqs[i, j * 4 + k]) + torch.log(kmer_1_vec[i, j])
    if rai_max == None:
        rai_max = torch.amax(rai)
    if rai_min == None:
        rai_min = torch.amin(rai)
    # print(f"max = {rai_max} min = {rai_min}")
    rai = (rai - rai_min) / (rai_max - rai_min)
    return rai, rai_min, rai_max

def load_reads_gan(in_dir, read_size=100, use_rev=False, label_encoder=None):
    all_reads = read_fastas_from_dir_gan(in_dir, read_size, use_rev=use_rev, label_encoder=label_encoder)
    read_dataset = ReadsDataset(all_reads)
    return read_dataset

def read_fastas_from_dir_perceiver(in_dir, read_size, kmer_size, step_size, use_rev=True, label=None,
                                   label_encoder=None, kmer_dict=None):
    all_reads, all_labels = [], []
    all_nucs = ["A", "C", "T", "G"]
    if kmer_dict == None:
        all_kmers = generate_kmers(kmer_size)
        kmer_dict = {}
        for i, tmp_kmer in enumerate(all_kmers):
            kmer_dict[tmp_kmer] = i
    for count, in_file in enumerate(glob.glob(f"{in_dir}/*.fa")):
        print(f"reading {in_file}")
        got_labels = False
        for record in SeqIO.parse(in_file, "fasta"):
            if not got_labels:
                if label == None and label_encoder == None:
                    labels = [int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]]
                elif label == None and label_encoder != None:
                    labels = label_encoder[",".join(record.description.split(":")[-1].split(" ")[0].split(";")[0:6])]
                else:
                    labels = [count]
                got_labels = True
            tmp_nuc_arr = []
            for_seq = re.sub(r'[N]', random.choice(all_nucs), str(record.seq.upper()))
            
            length = len(for_seq)
            if length < read_size:
                continue
            # seq and rev_comp
            tmp_for = []
            for j in range(0, read_size - kmer_size + 1, step_size):
                #try:
                tmp_for.append(kmer_dict[for_seq[j:j+kmer_size]])
                #except:
                #    tmp_for.append(nuc_dict[random.choice(all_nucs)])
            #tmp_for = for_seq
            tmp_nuc_arr.extend(tmp_for)
            if use_rev:
                rev_seq = re.sub(r'[N]', random.choice(all_nucs), str(Seq.reverse_complement(record.seq).upper()))
                #tmp_rev = rev_seq
                tmp_rev = []
                for j in range(0, read_size - kmer_size + 1, step_size):
                    #try:
                    tmp_rev.append(kmer_dict[rev_seq[j:j+kmer_size]])
                    #except:
                    #    tmp_rev.append(nuc_dict[random.choice(all_nucs)])
                tmp_nuc_arr.extend(tmp_rev)
            all_reads.append(tmp_nuc_arr)
            all_labels.append(labels)
            #print(f"tmp_nuc_arr = {tmp_nuc_arr}\ntmp_arr = {tmp_arr}")
            #exit()
    return np.array(all_reads), np.array(all_labels), kmer_dict

def test_perceiver(model, test_dataloader, criterion, device, args, num_classes):
    with torch.no_grad():
        preds = []
        truths = []
        round_preds = []
        total_test_loss = 0.0
        print(f"len test_dataloader = {len(test_dataloader)}")
        for (data, labels) in test_dataloader:
            data, labels = data.to(device), labels.to(device)
            if args.debug:
                print(f"data = {data}\nlabels = {labels}")
            pred = model(data)
            if args.debug:
                print(f"pred = {pred}")
            #loss = criterion(torch.log(pred), labels)
            loss = criterion(torch.log(pred), labels)
            total_test_loss += float(loss.item())
            round_pred = torch.argmax(pred, dim=1).cpu().numpy()
            # print(f"round_pred = {round_pred}")
            round_preds.extend(round_pred)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())
        # print(f"rounds = {round_preds}\npreds = {preds}\ntruths = {truths}")
        truths = np.array(truths)
        print(f"truths shape = {truths.shape}")
        print(f"unique truths = {np.unique(truths)}\nunique round_preds = {np.unique(round_preds)}")
        acc = accuracy_score(truths, round_preds)
        f1 = f1_score(truths, round_preds, average="macro")
        prec = precision_score(truths, round_preds, average="macro")
        rec = recall_score(truths, round_preds, average="macro")
        """
        # Convert truths to 2d array for AUROC
        truths = np.array(truths)
        truths_2d = np.zeros((truths.size, truths.max() + 1))
        print(f"truths_2d size = {truths_2d.shape}")
        truths_2d[np.arange(truths.size), truths] = 1
        np.set_printoptions(threshold=np.inf)
        print(f"truths_2d = {truths_2d}\nunique truths_2d = {np.unique(truths_2d)}\ntruths_2d shape = {truths_2d.shape}\npreds len = {len(preds)}")
        preds_np = np.array(preds)
        print(f"preds_np shape = {preds_np.shape}")
        auprc = average_precision_score(truths_2d, preds_np)
        auroc = roc_auc_score(truths_2d, preds_np, multi_class="ovr")
        """
        test_loss = total_test_loss / len(test_dataloader)
        return test_loss, acc, f1, prec, rec
        #return test_loss, acc, f1, prec, rec, auprc, auroc

def read_fastas_from_dir_transformer(in_dir, read_size, use_rev=True, label=None, label_encoder=None):
    all_reads, all_labels = [], []
    nuc_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
    for count, in_file in enumerate(glob.glob(f"{in_dir}/*.fa")):
        print(f"reading {in_file}")
        got_labels = False
        for record in SeqIO.parse(in_file, "fasta"):
            if not got_labels:
                if label == None and label_encoder == None:
                    labels = [int(i) for i in record.description.split(":")[-1].split(" ")[0].split(";")[0:6]]
                elif label == None and label_encoder != None:
                    labels = label_encoder[",".join(record.description.split(":")[-1].split(" ")[0].split(";")[0:6])]
                else:
                    labels = [count]
                got_labels = True
            tmp_nuc_arr = []
            for_seq = str(record.seq.upper())
            length = len(for_seq)
            if length < read_size:
                continue
            # seq and rev_comp
            tmp_for = [nuc_dict[tmp_nuc] for tmp_nuc in for_seq[:read_size]]
            # tmp_for = for_seq
            tmp_nuc_arr.extend(tmp_for)
            if use_rev:
                rev_seq = str(Seq.reverse_complement(record.seq).upper())
                # tmp_rev = rev_seq
                tmp_rev = [nuc_dict[tmp_nuc] for tmp_nuc in rev_seq[:read_size]]
                tmp_nuc_arr.extend(tmp_rev)
            all_reads.append(tmp_nuc_arr)
            all_labels.append(labels)
            # print(f"tmp_nuc_arr = {tmp_nuc_arr}\ntmp_arr = {tmp_arr}")
            # exit()
    return np.array(all_reads), np.array(all_labels)

class ReadsDatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, all_reads, all_labels):
        self.num_reads = len(all_reads)
        self.labels = torch.LongTensor(all_labels)
        #self.nuc_toks = torch.FloatTensor([[0], [1], [2], [3]])
        #self.embedding = torch.nn.Embedding.from_pretrained(self.nuc_toks)
        #self.reads = self.embedding(torch.LongTensor(all_reads))
        self.reads = torch.LongTensor(all_reads)

    def __len__(self):
        return len(self.reads)

    def __getitem__(self, idx):
        curr_read = self.reads[idx]
        curr_label = self.labels[idx]
        return curr_read, curr_label
        # return self.embedding(torch.LongTensor(curr_read)), torch.LongTensor(curr_label)

def test_transformer(model, test_dataloader, criterion, device, args):
    with torch.no_grad():
        preds = []
        truths = []
        round_preds = []
        total_test_loss = 0.0
        for (data, labels) in test_dataloader:
            data, labels = data.transpose(0,1).to(device), labels.to(device)
            if args.debug:
                print(f"data = {data}\nlabels = {labels}")
            pred = model(data)
            if args.debug:
                print(f"pred = {pred}\nlog pred = {torch.log(pred)}")
            #loss = criterion(torch.log(pred), labels)
            loss = criterion(torch.log(pred), labels)
            total_test_loss += float(loss.item())
            round_pred = torch.argmax(pred, dim=1).cpu().numpy()
            # print(f"round_pred = {round_pred}")
            round_preds.extend(round_pred)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())
        # print(f"rounds = {round_preds}\npreds = {preds}\ntruths = {truths}")
        print(f"unique truths = {np.unique(truths)}\nunique round_preds = {np.unique(round_preds)}")
        acc = accuracy_score(truths, round_preds)
        f1 = f1_score(truths, round_preds, average="macro")
        prec = precision_score(truths, round_preds, average="macro")
        rec = recall_score(truths, round_preds, average="macro")
        # Convert truths to 2d array for AUROC
        truths = np.array(truths)
        truths_2d = np.zeros((truths.size, truths.max() + 1))
        print(f"truths_2d size = {truths_2d.shape}")
        truths_2d[np.arange(truths.size), truths] = 1
        print(f"truths_2d = {np.array2string(truths_2d)}\nunique truths_2d = {np.unique(truths_2d)}\ntruths_2d shape = {truths_2d.shape}\npreds len = {len(preds)}")
        auprc = average_precision_score(truths_2d, preds)
        auroc = roc_auc_score(truths_2d, preds)
        test_loss = total_test_loss / len(test_dataloader)
        return test_loss, acc, f1, prec, rec, auprc, auroc

def print_mem_usage(device):
    print(f"mem allocated = {torch.cuda.memory_allocated(device) / 1024}KB")
    print(f"max mem allocated = {torch.cuda.max_memory_allocated(device) / 1024}KB")
    print(f"mem reserved = {torch.cuda.memory_reserved(device) / 1024}KB")
    print(f"max mem reserved = {torch.cuda.max_memory_reserved(device) / 1024}KB")
    return

def load_embeddings(embed_file, kmer_size):
    return vocab.Vectors(f"{embed_file}")
