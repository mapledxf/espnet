#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
#/home/data/xfding/train_result/tts
out_dir=/home/zlj/dxf/train_result/tts
# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32        # numebr of parallel jobs
dumpdir=$out_dir/dump # directory to dump full features
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000        # sampling frequency
fmax=7600       # maximum frequency
fmin=80         # minimum frequency
n_mels=80       # number of mel basis
n_fft=2048      # number of fft points
n_shift=300     # number of shift points
win_length=1200 # window length

# silence part trimming related
trim_threshold=25 # (in decibels)
trim_win_length=1024
trim_shift_length=256
trim_min_silence=0.01

# config files
train_config=conf/train_pytorch_transformer+spkemb.yaml
#train_config=conf/tuning/train_fastspeech.v3.single.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


openslr_aishell=/home/data/xfding/dataset/asr/aishell/data_aishell
aishell_spk_info=/home/data/xfding/dataset/asr/aishell/resource_aishell/speaker.info
csmsc_data=/home/data/xfding/dataset/tts/csmsc/CSMSC
cmlr_data=/home/data/xfding/dataset/tts/cmlr

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
#    echo "stage 0: Data preparation"
#    echo "stage 0: Prepare csmsc"
#    local/data_prep.sh $csmsc_data $out_dir/data/csmsc
    # Downsample to fs from 48k
#    utils/data/resample_data_dir.sh ${fs} $out_dir/data/csmsc

#    echo "stage 0: Prepare CMLR"
#    local/cmlr_data_prep.sh \
#	    $cmlr_data \
#	    $out_dir/data/cmlr || exit 1;
#    utils/data/resample_data_dir.sh ${fs} $out_dir/data/cmlr/train

#    echo "stage 0: Prepare aishell"
#    local/aishell_data_prep.sh \
#	    $openslr_aishell \
#            $aishell_spk_info \
#	    $out_dir/data/aishell || exit 1;
#    for sets in train dev test; do
#    	utils/data/resample_data_dir.sh ${fs} $out_dir/data/aishell/$sets
#    done
   
    /home/zlj/dxf/vwm_data_prepare/aidatatang_data_prep.sh /home/zlj/Datasets/aidatatang_200zh $out_dir/data
#    echo "stage 0: combine data"
#    utils/combine_data.sh $out_dir/data/train \
#        $out_dir/data/aidatatang_200zh/train $out_dir/data/aidatatang_200zh/dev $out_dir/data/aidatatang_200zh/test || exit 1;

    utils/validate_data_dir.sh --no-feats $out_dir/data/train
    utils/validate_data_dir.sh --no-feats $out_dir/data/test
    utils/validate_data_dir.sh --no-feats $out_dir/data/dev
fi

train_set="train"
dev_set="dev"

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

#    n_total=$(wc -l < $out_dir/data/train/wav.scp)
#    echo total set:$n_total
#    n_dev=$(($n_total * 2 / 100))
#    n_train=$(($n_total - $n_dev))
#    echo train set:$n_train, dev set:$n_dev
    # make a dev set
#    utils/subset_data_dir.sh --last $out_dir/data/train $n_dev $out_dir/data/${dev_set}
#    utils/subset_data_dir.sh --first $out_dir/data/train $n_train $out_dir/data/${train_set}

    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    fbankdir=$out_dir/fbank
    for x in train dev test; do
        # Trim silence parts at the begining and the end of audio
        mkdir -p $out_dir/exp/trim_silence/${x}/figs  # avoid error
        trim_silence.sh --cmd "${train_cmd}" \
            --nj ${nj} \
            --fs ${fs} \
            --win_length ${trim_win_length} \
            --shift_length ${trim_shift_length} \
            --threshold ${trim_threshold} \
            --min_silence ${trim_min_silence} \
            $out_dir/data/${x} \
            $out_dir/exp/trim_silence/${x}

        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            $out_dir/data/${x} \
            $out_dir/exp/make_fbank/${x} \
            ${fbankdir}
    done

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:$out_dir/data/${train_set}/feats.scp $out_dir/data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        $out_dir/data/${train_set}/feats.scp $out_dir/data/${train_set}/cmvn.ark $out_dir/exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        $out_dir/data/${dev_set}/feats.scp $out_dir/data/${train_set}/cmvn.ark $out_dir/exp/dump_feats/dev ${feat_dt_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    dict=$out_dir/data/lang_phn/${train_set}_units.txt
    echo "dictionary: ${dict}"

    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    mkdir -p $out_dir/data/lang_phn/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for padding idx
    text2token.py -s 1 -n 1 --trans_type phn $out_dir/data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type phn \
         $out_dir/data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type phn \
         $out_dir/data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=$out_dir/mfcc
    vaddir=$out_dir/mfcc
    for name in ${train_set} ${dev_set}; do
        utils/copy_data_dir.sh $out_dir/data/${name} $out_dir/data/${name}_mfcc_16k
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj $nj --cmd "$train_cmd" \
            $out_dir/data/${name}_mfcc_16k \
	    $out_dir/exp/make_mfcc_16k \
	    ${mfccdir}
        utils/fix_data_dir.sh $out_dir/data/${name}_mfcc_16k
        sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
            $out_dir/data/${name}_mfcc_16k \
	        $out_dir/exp/make_vad \
	        ${vaddir}
        utils/fix_data_dir.sh $out_dir/data/${name}_mfcc_16k
    done

    # Check pretrained model existence
    nnet_dir=$out_dir/exp/xvector_nnet_1a
#    nnet_dir=/home/data/xfding/pretrained/tts/0008_sitw_v2_1a/exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
    	download_from_google_drive.sh https://drive.google.com/open?id=1j1aEgWDVDTiFftTqVK6NbqIV2aUCn02d ${out_dir}/exp ".tar.gz"
    fi
    # Extract x-vector
    for name in ${train_set} ${dev_set}; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
            ${nnet_dir} \
	    $out_dir/data/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done
for name in ${train_set} ${dev_set}; do
    local/update_json.sh ${dumpdir}/${name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=$out_dir/exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/results \
           --tensorboard-dir $out_dir/tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

echo "Finished."
