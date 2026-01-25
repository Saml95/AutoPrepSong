import sys
from pathlib import Path
from tqdm import tqdm
import glob

if __name__=='__main__':
    origin_scp, output_path, fail_scp = sys.argv[1:]


    success_list = glob.glob(f"{output_path}/**/*.lyric.json", recursive=True)
    success_list = [Path(i).name.replace(".lyric.json", "") for i in success_list]
    lines = open(origin_scp).readlines()
    first_fail = open(fail_scp, 'w')
    for l in tqdm(lines):
        if Path(l.strip()).stem in success_list:
            continue
        else:
            first_fail.write(l)