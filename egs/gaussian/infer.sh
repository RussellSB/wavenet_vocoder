#!/bin/bash

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
VOC_DIR=$script_dir/../../
dumpdir=dump

# waveform global gain normalization scale
global_gain_scale=0.55

stage=1
stop_stage=2

# Leave empty to use latest checkpoint
eval_checkpoint=

# exp tag
tag="" # tag for managing experiments.

. $VOC_DIR/utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# exp name
if [ -z ${tag} ]; then
    expname=${spk}_train_no_dev_$(basename ${hparams%.*})
else
    expname=${spk}_train_no_dev_${tag}
fi
expdir=exp/$expname

feat_typ="logmelspectrogram"

# Input directories
data_root=../../../voice_conversion/src/out_infer/${inferdir}/gen
dump_org_train=$dumpdir/$spk/$feat_typ/org 

# Output directories                  
dump_org_dir=$dumpdir/${spk}_${inferdir}/$feat_typ/org   # extracted features (pair of <wave, feats>)
dump_norm_dir=$dumpdir/${spk}_${inferdir}/$feat_typ/norm # extracted features (pair of <wave, feats>)
outdir=out/${spk}_${inferdir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    python $VOC_DIR/preprocess.py wavallin $data_root ${dump_org_dir} \
            --hparams="global_gain_scale=${global_gain_scale}" --preset=$hparams

    # Apply normalization from training mean-var normalization stats
    python $VOC_DIR/preprocess_normalize.py ${dump_org_dir} $dump_norm_dir \
        $dump_org_train/meanvar.joblib
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Synthesis waveform from WaveNet"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    CUDA_VISIBLE_DEVICES="0,1" python $VOC_DIR/evaluate.py $dump_norm_dir $eval_checkpoint $outdir \
            --preset $hparams --hparams="batch_size=32, num_workers=0"
fi