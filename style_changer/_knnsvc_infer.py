"""kNN-SVC batch inference subprocess, run with cwd = the kNN-SVC repo.
Args: src_root tgt_root converted_dir ckpt_dir ckpt_type post_opt
"""
import os
import sys

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.getcwd())


def _load(path, normalize=True, **kw):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T).contiguous(), sr


torchaudio.load = _load

src_root, tgt_root, converted_dir, ckpt_dir, ckpt_type, post_opt = sys.argv[1:7]
device = os.environ.get("STYLE_DEVICE", "cpu")

from ddsp_hubconf import knn_vc

knn = knn_vc(pretrained=True, prematched=True, device=device,
             ckpt_type=ckpt_type, local_ckpt_dir=ckpt_dir)
knn.bulk_match(
    src_dataset_path=src_root, tgt_dataset_path=tgt_root, converted_audio_dir=converted_dir,
    topk=4, device=device, prioritize_f0=True, ckpt_type=ckpt_type, tgt_loudness_db=-16,
    required_subset_file=None, post_opt=post_opt,
)
