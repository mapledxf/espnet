#!/bin/bash
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

txt=/home/xfding/espnet/egs/vwm/tts1/tts.txt
# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=4
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=0
verbose=1      # verbose option

# feature configuration
fs=22050      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

trans_type="phn"

# embedding related
input_wav=/data/xfding/share/ASR/aidatatang_200zh/corpus/dev/G0002/T0055G0002S0289.wav

# decoding related
dict=/data/xfding/train_result/tts/data/lang_phn/train_units.txt
synth_model=/data/xfding/train_result/tts/exp/train_pytorch_train_pytorch_transformer/results/model.avg.best
decode_config=/home/xfding/espnet/egs/vwm/tts1/conf/decode.yaml
#dict=/home/zlj/dxf/espnet/egs/csmsc/tts1/decode/download/csmsc.transformer.v1/data/lang_phn/train_no_dev_units.txt
#synth_model=/home/zlj/dxf/train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best
#decode_config=/home/zlj/dxf/train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/decode.yaml

decode_dir=decode
griffin_lim_iters=64

. utils/parse_options.sh || exit 1;

synth_json=$(basename ${synth_model})
model_json="$(dirname ${synth_model})/${synth_json%%.*}.json"
use_speaker_embedding=$(grep use_speaker_embedding ${model_json} | sed -e "s/.*: \(.*\),/\1/")
if [ "${use_speaker_embedding}" = "false" ] || [ "${use_speaker_embedding}" = "0" ]; then
    use_input_wav=false
else
    use_input_wav=true
fi

base=$(basename $txt .txt)
decode_dir=${decode_dir}/${base}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p ${decode_dir}/data
    echo "$base X" > ${decode_dir}/data/wav.scp
    echo "X $base" > ${decode_dir}/data/spk2utt
    echo "$base X" > ${decode_dir}/data/utt2spk
    echo -n "$base " > ${decode_dir}/data/text
    cat $txt >> ${decode_dir}/data/text

    mkdir -p ${decode_dir}/dump
    data2json.sh --trans_type ${trans_type} ${decode_dir}/data ${dict} > ${decode_dir}/dump/data.json
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ${use_input_wav}; then
    echo "stage 1: x-vector extraction"

    utils/copy_data_dir.sh ${decode_dir}/data ${decode_dir}/data2
    echo "$base ${input_wav}" > ${decode_dir}/data2/wav.scp
    utils/data/resample_data_dir.sh 16000 ${decode_dir}/data2
    # shellcheck disable=SC2154
    steps/make_mfcc.sh \
        --write-utt2num-frames true \
        --mfcc-config conf/mfcc.conf \
        --nj 1 --cmd "$train_cmd" \
        ${decode_dir}/data2 ${decode_dir}/log ${decode_dir}/mfcc
    utils/fix_data_dir.sh ${decode_dir}/data2
    sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
        ${decode_dir}/data2 ${decode_dir}/log ${decode_dir}/mfcc
    utils/fix_data_dir.sh ${decode_dir}/data2

    nnet_dir=/data/xfding/pretrained_model/xvector_nnet_1a
#    nnet_dir=/home/zlj/dxf/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a $neet_dir
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi

    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 1 \
        ${nnet_dir} ${decode_dir}/data2 \
        ${decode_dir}/xvectors

    local/update_json.sh ${decode_dir}/dump/data.json ${decode_dir}/xvectors/xvector.scp
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Decoding"

    # shellcheck disable=SC2154
    ${decode_cmd} ${decode_dir}/log/decode.log \
        tts_decode.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --debugmode ${debugmode} \
        --verbose ${verbose} \
        --out ${decode_dir}/outputs/feats \
        --json ${decode_dir}/dump/data.json \
        --model ${synth_model}
fi

outdir=${decode_dir}/outputs; mkdir -p ${outdir}_denorm


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Synthesis with Neural Vocoder"
    dst_dir=${decode_dir}/wav_wnv

    # This is hardcoded for now.
    checkpoint=/data/xfding/pretrained_model/csmsc.parallel_wavegan.v1/checkpoint-400000steps.pkl
    parallel-wavegan-decode \
        --scp "${outdir}/feats.scp" \
        --checkpoint "${checkpoint}" \
        --outdir "${dst_dir}" \
        --verbose ${verbose}

    echo ""
    echo "Synthesized wav: ${decode_dir}/wav_wnv/${base}.wav"
    echo ""
    echo "Finished"
fi

