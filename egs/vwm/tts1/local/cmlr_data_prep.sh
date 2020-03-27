#!/bin/bash

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/data_aishell data/aishell"
  exit 1;
fi

. ./path.sh || exit 1;

cmlr_dir=$1
data=$2
train_dir=$data/local/train
mkdir -p $train_dir

echo "**** Creating cmlr data folder ****"
# find wav audio file for train, dev and test resp.
find $cmlr_dir -iname "*.wav" > $train_dir/wav.flist
n=`cat $train_dir/wav.flist | wc -l`
echo Found $n data files
echo Preparing $train_dir transcriptions

sed -e 's/\.wav//' $train_dir/wav.flist | awk -F '/' '{i=NF-2;printf("%s_%s\n",$i,$NF)}' > $train_dir/utt.list
sed -e 's/\.wav//' $train_dir/wav.flist | awk -F '/' '{i=NF-2;printf("%s\n",$i)}' >$train_dir/spk.list

mkdir -p $data/train

while read line; do
  tn=`dirname $line`/`basename $line .wav`.txt;
  head -n 1 $tn | awk '{{printf $0}}'; echo;
done < $train_dir/wav.flist > $train_dir/text.list
paste -d' ' $train_dir/utt.list $train_dir/text.list > $train_dir/trans.txt
local/clean_text_mandarin.py $train_dir/trans.txt phn | sort -u > $data/train/text

paste -d' ' $train_dir/utt.list $train_dir/wav.flist | sort > $data/train/wav.scp
paste -d' ' $train_dir/utt.list $train_dir/spk.list | sort > $data/train/utt2spk
utils/utt2spk_to_spk2utt.pl $data/train/utt2spk | sort > $data/train/spk2utt
echo "s1 m" > $data/train/spk2gender

utils/data/validate_data_dir.sh --no-feats $data/train || exit 1;

