
# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
from typing import Union, Dict
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import argparse

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(current_dir, 'thirdparty/music_Source_Separation_Training'))

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, validate_sndfile_subtype
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings

warnings.filterwarnings("ignore")

def format_filename(template, **kwargs):
    '''
    Formats a filename from a template. e.g "{file_name}/{instr}"
    Using slashes ('/') in template will result in directories being created
    Returns [dirnames, fname], i.e. an array of dir names and a single file name
    '''
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    *dirnames, fname = result.split("/")
    return dirnames, fname



def process(path):
    print(f"Processing track: {path}")
    try:
        mix, sr = librosa.load(path, sr=sample_rate, mono=False)
    except Exception as e:
        print(f'Cannot read track: {format(path)}')
        print(f'Error message: {str(e)}')
        return 

    # If mono audio we must adjust it depending on model
    if len(mix.shape) == 1:
        mix = np.expand_dims(mix, axis=0)
        if 'num_channels' in config.audio:
            if config.audio['num_channels'] == 2:
                print(f'Convert mono track to stereo...')
                mix = np.concatenate([mix, mix], axis=0)

    mix_orig = mix.copy()
    if 'normalize' in config.inference:
        if config.inference['normalize'] is True:
            mix, norm_params = normalize_audio(mix)

    waveforms_orig = demix(config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar)
    if args.use_tta:
        waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

    if args.extract_instrumental:
        instr = 'vocals' if 'vocals' in instruments else instruments[0]
        waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
        if 'instrumental' not in instruments:
            instruments.append('instrumental')

    file_name = os.path.basename(path)# 保留后缀

    for instr in instruments:
        estimates = waveforms_orig[instr]
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                estimates = denormalize_audio(estimates, norm_params)

        codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
        subtype = args.pcm_type

        dirnames, fname = format_filename(
            args.filename_template,
            instr=instr,
            # start_time=int(start_time),
            file_name=file_name,
            dir_name=os.path.dirname(path),
            model_type=args.model_type,
            model=os.path.splitext(os.path.basename(args.start_check_point))[0]
        )

        output_dir = os.path.join(args.store_dir, *dirnames)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{fname}.{codec}")
        sf.write(output_path, estimates.T, sr, subtype=subtype)
        print("Wrote file:", output_path)
        if args.draw_spectro > 0:
            output_img_path = os.path.join(output_dir, f"{fname}.jpg")
            draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
            print("Wrote file:", output_img_path)



def parse_args_inference(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for inference configuration.

    Builds the CLI for model selection, configuration path, input/output handling,
    device/runtime options, test-time augmentation, and optional LoRA checkpoints.
    If `dict_args` is provided, its key–value pairs override or supply CLI options
    programmatically; otherwise, arguments are read from `sys.argv`.

    Args:
        dict_args (Union[Dict, None]): Optional mapping of argument names to values
            used to override or supply CLI options programmatically.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all inference
        configuration values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", type=str, default="", help="path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="Code will generate spectrograms for resulted stems."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true',
                        help="invert vocals to get instrumental if provided")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24', 'FLOAT'], default='FLOAT',
                        help="PCM type for FLAC files (PCM_16 or PCM_24)")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--filename_template", type=str, default='{file_name}/{instr}',
                        help="Output filename template, without extension, using '/' for subdirectories. Default: '{file_name}/{instr}'")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()
    args.pcm_type = validate_sndfile_subtype(args)

    return args

########################

args = parse_args_inference(None)
device = "cpu"
if args.force_cpu:
    device = "cpu"
elif torch.cuda.is_available():
    print('CUDA is available, use --force_cpu to disable it.')
    device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
elif torch.backends.mps.is_available():
    device = "mps"

print("Using device: ", device)

model_load_start_time = time.time()
torch.backends.cudnn.benchmark = True

model, config = get_model_from_config(args.model_type, args.config_path)
if 'model_type' in config.training:
    args.model_type = config.training.model_type
if args.start_check_point:
    checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
    load_start_checkpoint(args, model, checkpoint, type_='inference')

print("Instruments: {}".format(config.training.instruments))

# in case multiple CUDA GPUs are used and --device_ids arg is passed
if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
    model = nn.DataParallel(model, device_ids=args.device_ids)

model = model.to(device)

print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

model.eval()

mixture_paths = open(args.input_folder, 'r').readlines()
mixture_paths = [line.strip() for line in mixture_paths]
sample_rate = getattr(config.audio, 'sample_rate', 44100)

print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")
instruments = prefer_target_instrument(config)[:]
os.makedirs(args.store_dir, exist_ok=True)

verbose = True
if not verbose:
    mixture_paths = tqdm(mixture_paths, desc="Total progress")

if args.disable_detailed_pbar:
    detailed_pbar = False
else:
    detailed_pbar = True

for path in mixture_paths:
    process(path)