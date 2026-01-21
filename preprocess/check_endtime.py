import json
from pathlib import Path
from tqdm import tqdm

SCP_PATH = "/mnt/conversationhubhot/yaoyaochang/speech/data/music/muse20260112/Muse_jsons.scp"
TOLERANCE = 0.1  # 允许的时间误差（秒）

def check_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"JSON load error: {e}"

    if "audio_length" not in data:
        return False, "Missing key: audio_length"

    if "segments" not in data or not data["segments"]:
        return False, "Missing or empty segments"

    audio_length = data["audio_length"]
    last_end = data["segments"][-1].get("end", None)

    if last_end is None:
        return False, "Last segment missing 'end'"

    diff = abs(audio_length - last_end)
    if diff > TOLERANCE:
        return False, f"Mismatch: audio_length={audio_length}, last_end={last_end}, diff={diff:.6f}"

    return True, None


def main():
    scp_path = Path(SCP_PATH)
    assert scp_path.exists(), f"SCP file not found: {scp_path}"

    total = 0
    mismatch = 0

    with open(scp_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines)):
            line = line.strip()
            if not line:
                continue

            total += 1
            json_path = Path(line)

            if not json_path.exists():
                print(f"[MISSING FILE] {json_path}")
                mismatch += 1
                continue

            ok, msg = check_json(json_path)
            if not ok:
                print(f"[ERROR] {json_path}: {msg}")
                mismatch += 1

    print("\n====== Summary ======")
    print(f"Total checked : {total}")
    print(f"Mismatched    : {mismatch}")
    print(f"OK            : {total - mismatch}")


if __name__ == "__main__":
    main()
