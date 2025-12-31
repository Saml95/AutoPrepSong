
# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(current_dir, 'thirdparty/music_Source_Separation_Training'))

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
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

    file_name = os.path.splitext(os.path.basename(path))[0]

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