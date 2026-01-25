## Set up environment

3.10 / 3.12 测试过ok

```
bash prepare.sh
```

## RUN

```
bash exec.sh
```

或者

```
python3 scripts/autoprep_song.py cfg_file=config/prep_luoxue.yaml
```

INPUT_SCP: 格式参照/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/luoxue_20251226_all.scp

包含wav和lrc文件，文件名需一致

其他各保存目录参见exec.sh

scripts/preprocess_scp.py 和 scripts/postprocess_combine_all.py 分别负责把输入scp里面的lyric和wav做match以及把各流程输出汇集到对应的json中。

scripts/postprocess_combine_all.py 也包括按规则过滤歌词等步骤，这俩CPU上提交任务之前弄好就行了

歌词过滤规则：

    - 小于0.2s的片段，以及时间戳比前面小的片段(一般是翻译)->丢弃

    - 片段间间隔小于1.5s，把上一个的结尾延长；

    - 。。。大于1.5s，加一个空白

    - 歌词与结构按覆盖的最大比例进行match

主函数包括run_struct_anal.py / run_separation_new.py / run_sentence_vad.py这仨，此外还有一些下载预训练参数等也在exec.sh里面，直接执行一下即可

## 流程（auto_prep.py)

### 1. Load .lrc文件 + GPT洗歌词

### 2. SongFormer 结构预测

### 3. bs_roformer 声伴分离

### 4. VAD 算end time 

### 4.5 歌词再过滤（语速/规则/时间戳调整）

### 5. 后处理

    - 根据上下文对结构进行纠错

    - 根据 WER去掉错误率太高的 （大概是这样，我感觉WER只能用来过滤 不能用来做label）

### 6. Diarization

### 7. merge

## 单步运行 run_struct_anal.py / run_separation_new.py / run_sentence_vad.py

理论上只需要把所有arg / argv 写死在.py里面，就能直接调用process函数

- run_struct_anal.py / run_separation_new.py:

```
def process(audio_path):
	...
	# save file on corresponding dir
	return None
```

- run_sentence_vad.py

```
def process(audio_path, lyric_path):
	...
	return [{"text":, "start":, "end":}, {...}, ...}
	# 需要手动保存
```

## AMLT 提交

```
cd amlt_submit/yamls/music_jwyu
amlt run amlt_luoxue_20251126.yaml
```
