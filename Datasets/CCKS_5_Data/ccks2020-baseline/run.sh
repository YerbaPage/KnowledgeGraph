#/bin/bsub> BSUB -J Dan
#BSUB -e /nfsshare/home/dl08/Dan/CCKS_5_Data/ccks2020-baseline/log/Job_%J.err
#BSUB -o /nfsshare/home/dl08/Dan/CCKS_5_Data/ccks2020-baseline/log/Job_%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -R "select [ngpus>0] rusage [ngpus_excl_p=1]"
python ccks2020_baseline.py > out.txt
