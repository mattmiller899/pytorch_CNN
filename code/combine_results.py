import sys
import os
import glob
import subprocess

results_dir = sys.argv[1]
out_file = sys.argv[2]

with open(out_file, "w") as out:
    out.write("ID,Kmer,NC,FC,Epochs,Embeds,F1_Avg,F1_Std,Prec_Avg,Prec_Std,Rec_Avg,Rec_Std\n")
    for in_dir in glob.glob(f"{results_dir}/*"):
        #print(f"{in_dir}")
        for in_file in glob.glob(f"{in_dir}/r300*"):
            #print(f"{in_file}")
            lc = subprocess.Popen(['wc', '-l', in_file], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            #print(lc.stdout.readline())
            line_count = str(lc.stdout.readline()).split(' ')[0][2]
            #print(f"{line_count}")
            if line_count == '7':
                print("here")
                f = subprocess.Popen(['tail', '-2', in_file], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                avgs = str(f.stdout.readline())[2:-3].split(',')
                stds = str(f.stdout.readline())[2:-3].split(',')
                print(f"{avgs}\n{stds}")
                bn = os.path.basename(in_dir).split('_')
                try:
                    bn.remove("monokmer")
                except:
                    pass
                try:
                    bn.remove("multikmer")
                except:
                    pass
                try:
                    bn.remove("1000filts")
                except:
                    pass
                try:
                    bn.remove("no")
                except:
                    pass
                try:
                    bn.remove("n")
                except:
                    pass
                try:
                    bn.remove("3f")
                except:
                    pass
                try:
                    bn.remove("revfor")
                except:
                    pass
                #print(bn)
                #out.write(f"{'_'.join(bn[0:])},{bn[0]},{bn[1]},{bn[2]},{'_'.join(bn[3:])},{avgs[-3]},{stds[-3]},{avgs[-2]},{stds[-2]},{avgs[-1]},{stds[-1]}\n")
                out.write(f"{'_'.join(bn[0:])},{bn[0]},{bn[1]},{bn[2]},{bn[3]},{'_'.join(bn[4:])},{avgs[-3]},{stds[-3]},{avgs[-2]},{stds[-2]},{avgs[-1]},{stds[-1]}\n")
