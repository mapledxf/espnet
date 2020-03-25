trans_type = "phn"
dict_path = "/home/data/xfding/train_dataset/tts/downloads/csmsc-transformer/data/lang_phn/train_no_dev_units.txt"
model_path = "/home/data/xfding/train_result/tts/exp/csmsc_train_pytorch_train_pytorch_transformer.v1.single.csmsc-transformer/results/model.loss.best"
#model_path = "/home/data/xfding/train_dataset/tts/downloads/csmsc-transformer/exp/train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best"

vocoder_path = "/home/data/xfding/train_dataset/tts/downloads/pwg_task1/checkpoint-400000steps.pkl"
vocoder_conf = "/home/data/xfding/train_dataset/tts/downloads/pwg_task1/config.yml"


# add path
import sys
sys.path.append("espnet")

# define device
import torch
device = torch.device("cuda")

# define E2E-TTS model
from argparse import Namespace
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import
idim, odim, train_args = get_model_conf(model_path)
model_class = dynamic_import(train_args.model_module)
model = model_class(idim, odim, train_args)
torch_load(model_path, model)
model = model.eval().to(device)
inference_args = Namespace(**{"threshold": 0.5, "minlenratio": 0.0, "maxlenratio": 10.0})

# define neural vocoder
import yaml
from parallel_wavegan.models import ParallelWaveGANGenerator
with open(vocoder_conf) as f:
    config = yaml.load(f, Loader=yaml.Loader)
vocoder = ParallelWaveGANGenerator(**config["generator_params"])
vocoder.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"]["generator"])
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to(device)

# define text frontend
from pypinyin import pinyin, Style
from pypinyin.style._utils import get_initials, get_finals
with open(dict_path) as f:
    lines = f.readlines()
lines = [line.replace("\n", "").split(" ") for line in lines]
char_to_id = {c: int(i) for c, i in lines}
def frontend(text):
    """Clean text and then convert to id sequence."""
    text = pinyin(text, style=Style.TONE3)
    text = [c[0] for c in text]
    print(f"Cleaned text: {text}")
    idseq = []
    for x in text:
        c_init = get_initials(x, strict=True)
        c_final = get_finals(x, strict=True)
        for c in [c_init, c_final]:
            if len(c) == 0:
                continue
            if c not in char_to_id.keys():
                print(f"WARN: {c} is not included in dict.")
                idseq += [char_to_id["<unk>"]]
            else:
                idseq += [char_to_id[c]]
    idseq += [idim - 1]  # <eos>
    return torch.LongTensor(idseq).view(-1).to(device)

print("now ready to synthesize!")

import time
print("請用中文輸入您喜歡的句子!")
input_text = input()

with torch.no_grad():
    start = time.time()
    x = frontend(input_text)
    c, _, _ = model.inference(x, inference_args)
    z = torch.randn(1, 1, c.size(0) * config["hop_size"]).to(device)
    c = torch.nn.ReplicationPad1d(
        config["generator_params"]["aux_context_window"])(c.unsqueeze(0).transpose(2, 1))
    y = vocoder(z, c).view(-1)
rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
print(f"RTF = {rtf:5f}")

import numpy as np
from scipy.io.wavfile import write

write('test.wav', config["sampling_rate"], y.view(-1).cpu().numpy())

#from IPython.display import display, Audio
#display(Audio(y.view(-1).cpu().numpy(), rate=config["sampling_rate"]))



