#!/bin/bash

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
VOC_DIR=$script_dir/../../
dumpdir=dump

# waveform global gain normalization scale
global_gain_scale=0.55

stage=1
stop_stage=2

# Batch size at inference time.
inference_batch_size=32
# Leave empty to use latest checkpoint
eval_checkpoint=
# Max number of utts. for evaluation( for debugging)
eval_max_num_utt=1000000

# exp tag
tag="" # tag for managing experiments.

. $VOC_DIR/utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)

# exp name
if [ -z ${tag} ]; then
    expname=${spk}_${train_set}_$(basename ${hparams%.*})
else
    expname=${spk}_${train_set}_${tag}
fi
expdir=exp/$expname

feat_typ="logmelspectrogram"

# Output directories
data_root=data/$spk                        # train/dev/eval splitted data
dump_org_dir=$dumpdir/$spk/$feat_typ/org   # extracted features (pair of <wave, feats>)
dump_norm_dir=$dumpdir/$spk/$feat_typ/norm # extracted features (pair of <wave, feats>)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    for s in ${datasets[@]};
    do
      python $VOC_DIR/preprocess.py wavallin $data_root/$s ${dump_org_dir}/$s \
        --hparams="global_gain_scale=${global_gain_scale}" --preset=$hparams
    done

    # Compute mean-var normalization stats
    find $dump_org_dir/$train_set -type f -name "*feats.npy" > train_list.txt
    python $VOC_DIR/compute-meanvar-stats.py train_list.txt $dump_org_dir/meanvar.joblib
    rm -f train_list.txt

    # Apply normalization
    for s in ${datasets[@]};
    do
      python $VOC_DIR/preprocess_normalize.py ${dump_org_dir}/$s $dump_norm_dir/$s \
        $dump_org_dir/meanvar.joblib
    done
    cp -f $dump_org_dir/meanvar.joblib ${dump_norm_dir}/meanvar.joblib
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: WaveNet training"
    python $VOC_DIR/train.py --dump-root $dump_norm_dir --preset $hparams \
      --checkpoint-dir=$expdir \
      --log-event-path=tensorboard/${expname} \
      --checkpoint=${expdir}/checkpoint_latest.pth
fi