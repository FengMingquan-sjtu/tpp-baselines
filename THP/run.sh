device=3
data=data/mimic_clip/
batch=4
n_head=2
n_layers=2
d_model=8
d_rnn=8
d_inner=8
d_k=8
d_v=8
dropout=0.1
lr=1e-4
smooth=0.1
epoch=50
log=log.txt
is_test=1
target=61

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log -is_test $is_test -target $target
