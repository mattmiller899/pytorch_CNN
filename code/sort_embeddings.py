import os
import sys
import itertools

in_file = sys.argv[1]
out_file = sys.argv[2]
with open(in_file, "r") as f:
    _ = f.readline()
    l = f.readline()
    tmp_kmer = l.rstrip().split(" ")[0]
    kmer_size = len(tmp_kmer)
print(f"kmer_size = {kmer_size}")
all_kmers = [''.join(p) for p in itertools.product(["A", "C", "G", "T"], repeat=kmer_size)]
print(f"all_kmers = {all_kmers}")
kmer_dict = {}
with open(in_file, "r") as f:
    _ = f.readline()
    for l in f:
        larr = l.rstrip().split(" ")
        kmer_dict[larr[0]] = larr[1:]
with open(out_file, "w") as out:
    for k in all_kmers:
        out.write(f"{k} {' '.join(kmer_dict[k])}\n")
    
