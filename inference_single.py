


import json
import os
import sys
import re
import uuid
import tempfile
from argparse import ArgumentParser
from typing import Optional
import torchaudio
import torch
import os
import time
import torch

from funasr import AutoModel as ASRMODEL
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoModel, BitsAndBytesConfig
from speech_tokenizer.modeling_whisper import WhisperVQEncoder

sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder

audio_token_pattern = re.compile(r"<\|audio_(\d+)\|>")

from queue import Queue
from threading import Thread

class TokenStreamer:
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value

def initialize_fn(args):
    global audio_decoder, feature_extractor, whisper_model, glm_model, glm_tokenizer, device
    # GLM
    # 支持 int4 加载（按需）
    dtype = torch.bfloat16
    bnb_config = None
    if hasattr(args, "dtype") and args.dtype == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    glm_model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantization_config=bnb_config if bnb_config else None,
        device_map={"": 0}
    ).eval().to(device)
    # Flow & Hift
    audio_decoder = AudioDecoder(
        config_path=os.path.join(args.flow_path, "config.yaml"),
        flow_ckpt_path=os.path.join(args.flow_path, "flow.pt"),
        hift_ckpt_path=os.path.join(args.flow_path, "hift.pt"),
        device=device
    )
    # Speech tokenizer
    whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)

@torch.inference_mode()
def glm_generate_stream(all_inputs,convert_token,previous_convert_token, temperature, top_p, max_new_tokens, device, past_key_values=None,mode=0):

    print('all_inputs',len(all_inputs))
    print(all_inputs)
    all_inputs_emb = glm_tokenizer([all_inputs], return_tensors="pt").to(device)
    all_input_ids = all_inputs_emb['input_ids']
    all_input_position_ids = all_inputs_emb['position_ids']
    print('aaaaaaaaaaaa',all_input_ids.shape)
    print(all_input_ids)
    all_input_attention_mask = all_inputs_emb['attention_mask']
    input_attention_mask = all_input_attention_mask
    print('input_attention_mask',input_attention_mask)


    print('all_input_ids',all_input_ids.shape[1])
    if past_key_values is not None:
        print('past_key_values',past_key_values[0][0].shape[2])


    print('previous_convert_token',previous_convert_token)


    # if past_key_values is not None:
    #     past_length = past_key_values[0][0].shape[2]
    #     print(past_length)
    #     input_ids = all_input_ids[:, past_length:]
        # if convert_token is not None:
        #     input_position_ids = all_input_position_ids[:,start+convert_ids.shape[1]+(all_input_ids.shape[1]-end):start+convert_ids.shape[1]+(all_input_ids.shape[1]-end)+all_input_ids.shape[1]]
        # else:
        #     input_position_ids = all_input_position_ids[:, past_length:]
        
        # print('all_input_position_ids_after',all_input_position_ids)


    if convert_token is not None and mode == 0:
        print('convert_token',convert_token)
        
        convert_ids = glm_tokenizer([convert_token], return_tensors="pt").to(device)['input_ids'][:,2:]
        # print('convert_ids',convert_ids)
        # print('convert_ids', glm_tokenizer([convert_token], return_tensors="pt").to(device)['input_ids'])
        previous_convert_ids = glm_tokenizer([previous_convert_token], return_tensors="pt").to(device)['input_ids'][:,2:]
        print('previous_convert_ids',previous_convert_ids)

        def find_sublist_index(big_list, sub_list):
            """在big_list中查找sub_list的起止下标，返回(start, end)"""
            for i in range(len(big_list) - len(sub_list) + 1):
                if (big_list[i:i+len(sub_list)] == sub_list).all():
                    return i, i+len(sub_list)
            return None
        start,end = find_sublist_index(all_input_ids[0], previous_convert_ids[0])
        print('start',start)
        print('end',end)
        convert_position = all_input_position_ids[:, start:start+convert_ids.shape[1]]
        end_convert_ids = all_input_ids[:,end:]
        print('end_convert_ids.shape[0]',end_convert_ids.shape[0])
        endconvert_position = all_input_position_ids[:, start+convert_ids.shape[1]:start+convert_ids.shape[1]+end_convert_ids.shape[1]]
        print('111111111111111lengthbefore',all_input_ids.shape[1])
        
        print('attention_mask',input_attention_mask.shape[1])


    else:
        start = None
        end = None
        convert_ids = None
        endconvert_position = None
        convert_position = None
        end_convert_ids = None

    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2]
        print(past_length)
        input_ids = all_input_ids[:, past_length:]
        input_position_ids = all_input_position_ids[:, past_length:]
    else:
        input_ids = all_input_ids
        input_position_ids = all_input_position_ids

    streamer1 = TokenStreamer(skip_prompt=True)
    streamer = TokenStreamer(skip_prompt=True)

    if past_key_values is not None:
        print('aaa', type(past_key_values), len(past_key_values))
        print("第0层 ", past_key_values[0][0].shape)

    # print('input_ids',input_ids.shape)
    # print('attention_mask',inputs['attention_mask'].shape)


    

    # print('111convert_ids',convert_ids)
    # print('111end_convert_ids',end_convert_ids)
    # print('111convert_position',convert_position)
    # print('111endconvert_position',endconvert_position)
    # print('111delete_start',start)
    # print('111delete_end',end)


# 0,1,2,3,4,5

    if start is not None and mode == 0:
        print('333convert_ids', glm_tokenizer.decode(convert_ids.squeeze(0).tolist(), spaces_between_special_tokens=False))
        print('3333end_convert_ids', glm_tokenizer.decode(end_convert_ids.squeeze(0).tolist(), spaces_between_special_tokens=False))
        print('333input_ids',glm_tokenizer.decode(input_ids.squeeze(0).tolist(), spaces_between_special_tokens=False))
        print('111111111111',start+convert_ids.shape[1])
        length = all_input_attention_mask.shape[1]-input_ids.shape[1]-end
        gen_kwargs = dict(  
            input_ids=convert_ids,
            position_ids=convert_position,
            attention_mask=all_input_attention_mask[:,:start+convert_ids.shape[1]],
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            return_dict_in_generate=True,
            streamer=streamer1,
            convert_position=all_input_position_ids[:,end:end+length],
            endconvert_position=all_input_position_ids[:,start+convert_ids.shape[1]:start+convert_ids.shape[1]+length],
            delete_start=start,
            delete_end=end,
            past_key_values=past_key_values,
        )
        output = glm_model.generate(**gen_kwargs)
        new_layer = []
        for layer_idx, i in enumerate(output.past_key_values):
            new_i = []
            for kv_idx, j in enumerate(i):
                # print('55555',j.shape[2])
                # print('66666',length)
                # print(start + convert_ids.shape[1] + length)
                new_j = j[:, :, :start + convert_ids.shape[1] + length]
                new_i.append(new_j)
            new_layer.append(tuple(new_i))
        past_key_values = tuple(new_layer)
        input_position_ids = all_input_position_ids[:,start + convert_ids.shape[1] + length:start + convert_ids.shape[1] + length + input_ids.shape[1]]
        input_attention_mask = all_input_attention_mask[:,:start + convert_ids.shape[1] + length + input_ids.shape[1]]

        # print("模型生成的token ids:", output["sequences"] if "sequences" in output else output[0])
        # print("模型生成的文本:", glm_tokenizer.decode((output["sequences"][0] if "sequences" in output else output[0][0]).tolist(), spaces_between_special_tokens=False))

    print('4444444')

    gen_kwargs = dict(  
        input_ids=input_ids,
        position_ids=input_position_ids,
        attention_mask=input_attention_mask,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        return_dict_in_generate=True,
        streamer=streamer,
        past_key_values=past_key_values,
    )
    

    output_container = {}

    def run_generate():
        output = glm_model.generate(**gen_kwargs)
        print(output.keys())
        if hasattr(output, "past_key_values"):
            output_container["past_key_values"] = output.past_key_values
        elif isinstance(output, tuple) and len(output) > 1:
            output_container["past_key_values"] = output[1]

    thread = Thread(target=run_generate)
    thread.start()

    # output = glm_model.generate(**gen_kwargs)
    # print("模型生成的token ids:", output["sequences"] if "sequences" in output else output[0])
    # print("模型生成的文本:", glm_tokenizer.decode((output["sequences"][0] if "sequences" in output else output[0][0]).tolist(), spaces_between_special_tokens=False))


    for token_id in streamer:
        yield token_id
    yield output_container.get("past_key_values", None)


def inference_fn(
        temperature: float,
        top_p: float,
        max_new_token: int,
        input_mode,
        audio_path: Optional[str],
        input_text: Optional[str],
        history: list,
        previous_input_tokens: str,
        previous_completion_tokens: str,
        previous_kv_cache=None,
        save_history=1,
        mode=0
    ):
    def extract_user_round(text, i):
        # 找到所有 <|user|> 的位置
        user_positions = [m.start() for m in re.finditer(r'<\|user\|>', text)]
        if i >= len(user_positions):
            return None  # 超出范围
        start = user_positions[i]
        # 找下一个 <|user|> 的位置
        if i+1 < len(user_positions):
            end = user_positions[i+1]
        else:
            end = len(text)
        return text[start:end]

    def extract_history_round(history, i):
        start = i * 3
        end = start + 3
        if start >= len(history):
            return None
        return history[start:end]


    # 提取需要替换的文本
    current_turn = int(len(history)/3)
    # print('current_turn',current_turn)
    # 0,1,2,3,4,5,6,
    previous_convert_token = None
    convert_token = None

    if current_turn > save_history:
        convert_turn = current_turn - save_history - 1
        convert_history = extract_history_round(history,convert_turn)

        # 转译user text
        asr_model_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/huggingface.co/FunAudioLLM/SenseVoiceSmall"
        asrmodel = ASRMODEL(
            model=asr_model_dir,
            device="cuda:3",
        )
        res = asrmodel.generate(
                input=convert_history[0]['content']['path'],
                cache={},
                language="auto", 
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
        convert_token = '<|user|>\n' + res[0]["text"].split('>')[-1] + '<|assistant|>streaming_transcription\n' + convert_history[2]['content']

        previous_convert_token = extract_user_round(previous_input_tokens,convert_turn)

        # convert_token = previous_convert_token


    if input_mode == "audio":
        assert audio_path is not None
        history.append({"role": "user", "content": {"path": audio_path}})
        audio_tokens = extract_speech_token(
            whisper_model, feature_extractor, [audio_path]
        )[0]
        if len(audio_tokens) == 0:
            print("No audio tokens extracted")
            return history, previous_input_tokens, previous_completion_tokens, '', None
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
        user_input = audio_tokens
        system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
    else:
        assert input_text is not None
        history.append({"role": "user", "content": input_text})
        user_input = input_text
        system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
    

    # inputs传新的
    # attention mask传所有的
    # position ids传新的

    all_inputs = None
    
    all_inputs = previous_input_tokens + previous_completion_tokens
    if "<|system|>" not in all_inputs:
        all_inputs += f"<|system|>\n{system_prompt}"
    all_inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
    # all_inputs += final_str

    print(len(all_inputs))

    
    # tokenized = glm_tokenizer([inputs], return_tensors="pt")
    # input_ids = tokenized['input_ids'].to(device)


    # inputs = previous_input_tokens + previous_completion_tokens
    # inputs = inputs.strip()
    # if "<|system|>" not in inputs:
    #     inputs += f"<|system|>\n{system_prompt}"
    # inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

    text_tokens, audio_tokens = [], []
    audio_offset = glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
    end_token_id = glm_tokenizer.convert_tokens_to_ids('<|user|>')
    complete_tokens = []
    prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
    this_uuid = str(uuid.uuid4())
    tts_speechs = []
    tts_mels = []
    prev_mel = None
    is_finalize = False
    block_size = 10

    new_kv_cache = None

    gen = glm_generate_stream(
        all_inputs=all_inputs,  
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_token,
        device=device,
        past_key_values=previous_kv_cache,
        convert_token=convert_token,
        previous_convert_token=previous_convert_token,
        mode=mode
    )
    if convert_token is not None and mode == 0:
        all_inputs = all_inputs.replace(previous_convert_token, convert_token)

    for token_id in gen:
        if isinstance(token_id, (tuple, list)) or token_id is None:
            new_kv_cache = token_id
            # print('bbbb',new_kv_cache)
            break
        if token_id == end_token_id:
            is_finalize = True
        if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
            block_size = 20
            tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
            tts_speech, tts_mel = audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                          prompt_token=flow_prompt_speech_token.to(device),
                                                          prompt_feat=prompt_speech_feat.to(device),
                                                          finalize=is_finalize)
            prev_mel = tts_mel
            tts_speechs.append(tts_speech.squeeze())
            tts_mels.append(tts_mel)
            flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
            audio_tokens = []
        if not is_finalize:
            complete_tokens.append(token_id)
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)
            else:
                text_tokens.append(token_id)
    tts_speech = torch.cat(tts_speechs, dim=-1).cpu() if tts_speechs else None
    complete_text = glm_tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)
    # print('complete_text',complete_text)
    # 保存音频
    audio_file = None
    if tts_speech is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_file = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice'+f.name
            torchaudio.save(audio_file, tts_speech.unsqueeze(0), 22050, format="wav")
        print(f"[Assistant Audio Output Saved]: {audio_file}")
    # 输出文本
    # print(f"[Assistant Text Output]: {glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)}")
    history.append({"role": "assistant", "content": {"path": audio_file, "type": "audio/wav"}})
    history.append({"role": "assistant", "content": glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)})
    # print(history)
    return history, all_inputs, complete_text, audio_file, new_kv_cache

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--flow-path", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/huggingface.co/THUDM/glm-4-voice-decoder")
    parser.add_argument("--model-path", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/huggingface.co/THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", type= str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/huggingface.co/THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--mode", type=int, default=0)

    args = parser.parse_args()
    device = "cuda"
    # 初始化
    print("Initializing models, please wait ...")
    initialize_fn(args)
    print("Initialization done.")

    history = []
    input_tokens = ""
    completion_tokens = ""
    kv_cache = None

    audio_inputs = [
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/iclr2025-longcontext/audio0/tell me a joke.wav"
    ]
    
    # audio_inputs += [
    #     "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/input/你背一下出师表.wav"
    # ] 
    
    # audio_inputs += [
    #     "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/input/给我讲一个和羊排相关的故事.wav"
    # ] 
    # audio_inputs += [
    #     "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/input/请你说5遍，我爱学计算机.wav"
    # ] 

    # audio_inputs.append(
    #     "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/input/我的午餐吃了什么.wav"
    # )
    

    log_file = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/0609_0_6.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("GLM-4-Voice Inference Log\n")


    for i, audio_path in enumerate(audio_inputs):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- GLM-4-Voice Auto Chat Round {i+1} ---\n")
            print(f"\n--- GLM-4-Voice Auto Chat Round {i+1} ---")
            if not os.path.exists(audio_path):
                info = f"Audio file does not exist: {audio_path}\n"
                print(info)
                f.write(info)
                continue

            # 显存统计前清零
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()

            # 推理
            history, input_tokens, completion_tokens, audio_file, kv_cache = inference_fn(
                temperature=0.2,
                top_p=0.8,
                max_new_token=2000,
                input_mode="audio",
                audio_path=audio_path,
                input_text=None,
                history=history,
                previous_input_tokens=input_tokens,
                previous_completion_tokens=completion_tokens,
                previous_kv_cache=kv_cache,
                mode=args.mode
            )
            print('completion_tokens',completion_tokens)

            elapsed = time.time() - start_time
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

            f.write("-" * 40 + "\n")
            f.write(f"[Round {i+1}] Audio input: {audio_path}\n")
            if audio_file:
                f.write(f"[Round {i+1}] Audio output saved at: {audio_file}\n")
            for item in reversed(history):
                if item["role"] == "assistant" and isinstance(item["content"], str):
                    f.write(f"[Round {i+1}] Text output: {item['content']}\n")
                    break
            f.write(f"[Round {i+1}] Max GPU Memory Used: {max_mem:.2f} GB\n")
            f.write(f"[Round {i+1}] Inference Time: {elapsed:.2f} seconds\n")
            f.write("-" * 40 + "\n")

            print("-" * 40)
            print(f"[Round {i+1}] Audio input: {audio_path}")
            if audio_file:
                print(f"[Round {i+1}] Audio output saved at: {audio_file}")
            for item in reversed(history):
                if item["role"] == "assistant" and isinstance(item["content"], str):
                    print(f"[Round {i+1}] Text output: {item['content']}")
                    break
            # print('77777777777777',kv_cache[0][0].shape[2])
            print(f"[Round {i+1}] Max GPU Memory Used: {max_mem:.2f} GB")
            print(f"[Round {i+1}] Inference Time: {elapsed:.2f} seconds")
            print("-" * 40)




''' 
151331, 151333, 151335,    198,   1474,    686,   3410,    498,    448,
            264,   8805,   7600,     13,   3155,    432,   3019,    553,   3019,
             13,   5512,     11,   1744,    911,    279,   7600,    323,   5889,
            304,    264,  94171,   4141,  11561,     11,    448,    220,  99366,
           1467,   3950,   8109,    553,    220,  99916,   7699,  11206,     13,
            220, 151336,    198, 151343, 159290, 161774, 154713, 167499, 165821,
         167442, 154315, 166394, 157067, 154510, 159964, 154031, 166615, 153946,
         155765, 161126, 160014, 164425, 160014, 151344, 151337,   4027,    287,
           7964,   1453,    198
 151331, 151333, 151335,    198,   1474,    686,   3410,    498,    448,
            264,   8805,   7600,     13,   3155,    432,   3019,    553,   3019,
             13,   5512,     11,   1744,    911,    279,   7600,    323,   5889,
            304,    264,  94171,   4141,  11561,     11,    448,    220,  99366,
           1467,   3950,   8109,    553,    220,  99916,   7699,  11206,     13,
            220, 151336,    198, 151343, 159290, 161774, 154713, 167499, 165821,
         167442, 154315, 166394, 157067, 154510, 159964, 154031, 166615, 153946,
         155765, 161126, 160014, 164425, 160014, 151344, 151337,   4027,    287,
           7964,   1453,    198, 109377,   3837, 124675,   6313, 101665, 109213,
          99215,  99444,  99212,  11314, 166648, 164526, 163431, 155030, 159141,
         167785, 163012, 167510, 165067, 158729, 159316, 167061, 156487, 167785,
         152945, 162242, 157345, 155220, 156583, 159316, 166768, 166768, 167840,
         155444, 160466, 155342, 157515, 159530, 153961, 167551, 165123, 155721,
         152703, 154548, 165926, 163830, 155220, 152459, 155773, 161696, 156246,
         167188, 161373, 157635, 164078, 160166, 160014, 160014, 160014, 160526,
         151336,    198, 151343, 163168, 161774, 164482, 158553, 153642, 163492,
         166693, 166286, 156507, 158217, 158278, 153219, 160872, 163251, 167323,
         164639, 168492, 160789, 160703, 163482, 155017, 167393, 160166, 151344,
         151337,   4027,    287,   7964,   1453,    198
'''


# 0 - 编辑kv
# 1 - 有kv，不编辑
# 2 - 有kv，直接替换
# 3 - 没kv



'''
<|user|>
我的早餐吃了大米，午餐吃了豆浆，晚餐吃了羊排。<|assistant|>streaming_transcription
你的饮食挺丰富的哦！不过要注意营养均衡，尽量搭配一些蔬菜和水果。这样会让你的饮食更加健康哦！有什么我可以帮你的吗？


all_inputs <|system|>
User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. <|user|>
<|begin_of_audio|><|audio_14295|><|audio_1720|><|audio_15198|><|audio_15335|><|audio_10648|><|audio_12021|><|audio_1504|><|audio_14319|><|audio_4322|><|audio_14248|><|audio_15943|><|audio_12821|><|audio_13982|><|audio_219|><|audio_2186|><|audio_8406|><|audio_15769|><|audio_11811|><|audio_1678|><|audio_16353|><|audio_9403|><|audio_12329|><|audio_11725|><|audio_15040|><|audio_12072|><|audio_2082|><|audio_1558|><|audio_14831|><|audio_2487|><|audio_4535|><|audio_11922|><|audio_12852|><|audio_12751|><|audio_5368|><|audio_1317|><|audio_13783|><|audio_9807|><|audio_4610|><|audio_15086|><|audio_1357|><|audio_3703|><|audio_14047|><|audio_2367|><|audio_13513|><|audio_2573|><|audio_12072|><|audio_15040|><|audio_15411|><|audio_12072|><|audio_2710|><|audio_3044|><|audio_13021|><|audio_12013|><|audio_5232|><|audio_3283|><|audio_6199|><|audio_14582|><|audio_15033|><|audio_996|><|audio_2280|><|audio_3187|><|audio_370|><|audio_9481|><|audio_1696|><|audio_7727|><|audio_4255|><|audio_9450|><|audio_11296|><|audio_10045|><|end_of_audio|><|assistant|>streaming_transcription
你的饮食挺丰富的哦！不过要注意营养均衡，尽量搭配<|audio_10815|><|audio_9374|><|audio_10798|><|audio_10634|><|audio_13524|><|audio_11813|><|audio_1781|><|audio_6816|><|audio_8442|><|audio_9714|><|audio_15965|><|audio_16116|><|audio_11772|><|audio_606|><|audio_14692|><|audio_5287|><|audio_8561|><|audio_15528|><|audio_13941|><|audio_10055|><|audio_2401|><|audio_15809|><|audio_14032|><|audio_15510|><|audio_5492|><|audio_5492|>一些蔬菜和水果。这样会让你的饮食更加健康哦！<|audio_6963|><|audio_8979|><|audio_1272|><|audio_7859|><|audio_8510|><|audio_7548|><|audio_80|><|audio_12421|><|audio_1176|><|audio_1993|><|audio_15897|><|audio_3934|><|audio_1056|><|audio_7795|><|audio_512|><|audio_7331|><|audio_5195|><|audio_6199|><|audio_3586|><|audio_1730|><|audio_11516|><|audio_10345|><|audio_16141|><|audio_15470|><|audio_15312|><|audio_5492|>有什么我可以帮你的吗？<|audio_6376|><|audio_6963|><|audio_7384|><|audio_13992|><|audio_6962|><|audio_1480|><|audio_8563|><|audio_14118|><|audio_12584|><|audio_10423|><|audio_11332|><|audio_8629|><|audio_14674|><|audio_11723|><|audio_12073|><|audio_3749|><|audio_31|><|audio_7798|><|audio_9840|><|audio_15273|><|audio_6252|><|audio_16244|><|audio_14505|><|audio_14193|><|audio_6774|><|audio_3133|><|audio_11116|><|audio_5703|><|audio_3320|><|audio_2686|><|audio_2184|><|audio_8113|><|audio_14383|><|audio_5408|><|audio_6376|><|audio_3472|><|audio_3472|><|audio_6376|><|audio_15526|><|audio_12213|><|audio_9806|><|audio_1025|><|audio_8160|><|audio_13650|><|audio_7338|><|audio_4583|><|audio_13609|><|audio_5396|><|audio_6237|><|audio_10114|><|audio_1781|><|audio_6816|><|audio_8442|><|audio_12057|><|audio_14427|><|audio_14642|><|audio_15989|><|audio_7548|><|audio_11332|><|audio_1569|><|audio_11147|><|audio_6540|><|audio_15781|><|audio_10344|><|audio_5240|><|audio_649|><|audio_13466|><|audio_6572|><|audio_7892|><|audio_15510|><|audio_11143|><|audio_6376|><|audio_6963|><|audio_15487|><|audio_6583|><|audio_8113|><|audio_5518|><|audio_15935|><|audio_13537|><|audio_267|><|audio_12287|><|audio_15781|><|audio_3752|><|audio_2195|><|audio_13573|><|audio_699|><|audio_14184|><|audio_106|><|audio_3420|><|audio_9343|><|audio_5781|><|audio_11763|><|audio_9020|><|audio_5282|><|audio_11725|><|audio_7813|><|audio_7661|><|audio_8173|><|audio_12072|><|audio_9693|><|user|>
<|begin_of_audio|><|audio_14295|><|audio_7998|><|audio_2360|><|audio_8717|><|audio_15616|><|audio_2204|><|audio_176|><|audio_5783|><|audio_3792|><|audio_11346|><|audio_4904|><|audio_11123|><|audio_10401|><|audio_3960|><|audio_15918|><|audio_4849|><|audio_16076|><|audio_15503|><|audio_2710|><|audio_12072|><|audio_12072|><|audio_12072|><|audio_12072|><|audio_12072|><|audio_5411|><|audio_11795|><|audio_6730|><|audio_15029|><|audio_13067|><|audio_8288|><|audio_3793|><|audio_3246|><|audio_9036|><|audio_3161|><|audio_8581|><|audio_10654|><|audio_13546|><|audio_14636|><|audio_2591|><|audio_6191|><|audio_6963|><|audio_12072|><|audio_12072|><|audio_7661|><|end_of_audio|><|assistant|>streaming_transcription
我爱你，学计算机。我爱你，学计算机。我爱你，学<|audio_14295|><|audio_15146|><|audio_15870|><|audio_14052|><|audio_16238|><|audio_8175|><|audio_9551|><|audio_1289|><|audio_4176|><|audio_260|><|audio_6026|><|audio_2664|><|audio_6963|><|audio_3407|><|audio_2471|><|audio_3192|><|audio_10241|><|audio_3254|><|audio_4853|><|audio_11327|><|audio_4728|><|audio_14258|><|audio_11977|><|audio_4794|><|audio_5497|><|audio_3232|>计算机。我爱你，学计算机。我爱你，学计算机。<|audio_12714|><|audio_5492|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_2082|><|audio_16007|><|audio_14383|><|audio_14799|><|audio_8397|><|audio_11747|><|audio_6085|><|audio_2476|><|audio_13727|><|audio_260|><|audio_6026|><|audio_3547|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_15312|><|audio_5835|><|audio_2414|><|audio_6105|><|audio_11396|><|audio_5336|><|audio_15659|><|audio_8767|><|audio_3580|><|audio_8275|><|audio_4794|><|audio_5497|><|audio_9640|><|audio_14514|><|audio_11143|><|audio_7813|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_2082|><|audio_16007|><|audio_14383|><|audio_7552|><|audio_8397|><|audio_11747|><|audio_6085|><|audio_2476|><|audio_13727|><|audio_260|><|audio_6026|><|audio_3547|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_6963|><|audio_12476|><|audio_13532|><|audio_3192|><|audio_10241|><|audio_3254|><|audio_1802|><|audio_11327|><|audio_4728|><|audio_14258|><|audio_7506|><|audio_4794|><|audio_5497|><|audio_3232|><|audio_10270|><|audio_7813|><|audio_7661|><|audio_8173|><|audio_7661|><|audio_12072|><|audio_12072|><|audio_2066|><|audio_14415|><|audio_15146|><|audio_3044|><|audio_13178|><|audio_14799|><|audio_8397|><|audio_11747|><|audio_6085|><|audio_2476|><|audio_13727|><|audio_260|><|audio_6026|><|audio_3547|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_6963|><|audio_12476|><|audio_13532|><|audio_3192|><|audio_10241|><|audio_3254|><|audio_1802|><|audio_11327|><|audio_4728|><|audio_14258|><|audio_7506|><|audio_4794|><|audio_5497|><|audio_3232|><|audio_10270|><|audio_7813|><|audio_7661|><|audio_12072|><|audio_12072|><|audio_12072|><|user|>
<|begin_of_audio|><|audio_14295|><|audio_7998|><|audio_2360|><|audio_8717|><|audio_15616|><|audio_2204|><|audio_176|><|audio_5783|><|audio_3792|><|audio_11346|><|audio_4904|><|audio_11123|><|audio_10401|><|audio_3960|><|audio_15918|><|audio_4849|><|audio_16076|><|audio_15503|><|audio_2710|><|audio_12072|><|audio_12072|><|audio_12072|><|audio_12072|><|audio_12072|><|audio_5411|><|audio_11795|><|audio_6730|><|audio_15029|><|audio_13067|><|audio_8288|><|audio_3793|><|audio_3246|><|audio_9036|><|audio_3161|><|audio_8581|><|audio_10654|><|audio_13546|><|audio_14636|><|audio_2591|><|audio_6191|><|audio_6963|><|audio_12072|><|audio_12072|><|audio_7661|><|end_of_audio|><|assistant|>streaming_transcription
我爱你，学计算机。我爱你，学计算机。我爱你，学<|audio_14295|><|audio_15146|><|audio_15870|><|audio_14052|><|audio_16238|><|audio_8175|><|audio_9551|><|audio_1289|><|audio_4176|><|audio_260|><|audio_6026|><|audio_2664|><|audio_6963|><|audio_3407|><|audio_2471|><|audio_3192|><|audio_10241|><|audio_3254|><|audio_4853|><|audio_11327|><|audio_4728|><|audio_14258|><|audio_7506|><|audio_4794|><|audio_5497|><|audio_3232|>计算机。我爱你，学计算机。我爱你，学计算机。<|audio_15510|><|audio_3039|><|audio_7813|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_2082|><|audio_16007|><|audio_14383|><|audio_7552|><|audio_16244|><|audio_11747|><|audio_6085|><|audio_2476|><|audio_13727|><|audio_260|><|audio_6026|><|audio_3547|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_6963|><|audio_12476|><|audio_11015|><|audio_3793|><|audio_12060|><|audio_3254|><|audio_4853|><|audio_11327|><|audio_4728|><|audio_14258|><|audio_7506|><|audio_4794|><|audio_5497|><|audio_3232|><|audio_10270|><|audio_3039|><|audio_7813|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_14415|><|audio_2082|><|audio_3493|><|audio_6730|><|audio_316|><|audio_16244|><|audio_11747|><|audio_6085|><|audio_2476|><|audio_13727|><|audio_260|><|audio_6026|><|audio_3547|><|audio_12217|><|audio_12217|><|audio_12217|><|audio_6963|><|audio_14708|><|audio_11015|><|audio_7458|><|audio_12060|><|audio_2594|><|audio_9921|><|audio_3514|><|audio_7449|><|audio_1016|><|audio_1357|><|audio_4794|><|audio_5497|><|audio_3232|><|audio_10270|><|audio_7813|><|audio_8173|><|audio_7661|><|audio_12072|><|audio_12072|><|user|>
<|begin_of_audio|><|audio_14295|><|audio_7998|><|audio_2360|><|audio_8717|><|audio_15616|><|audio_2204|><|audio_176|><|audio_5783|><|audio_3792|><|audio_11346|><|audio_4904|><|audio_11123|><|audio_10401|><|audio_3960|><|audio_15918|><|audio_4849|><|audio_16076|><|audio_15503|><|audio_2710|><|audio_12072|><|audio_12072|><|audio_12072|><|audio_12072|><|audio_12072|><|audio_5411|><|audio_11795|><|audio_6730|><|audio_15029|><|audio_13067|><|audio_8288|><|audio_3793|><|audio_3246|><|audio_9036|><|audio_3161|><|audio_8581|><|audio_10654|><|audio_13546|><|audio_14636|><|audio_2591|><|audio_6191|><|audio_6963|><|audio_12072|><|audio_12072|><|audio_7661|><|end_of_audio|><|assistant|>streaming_transcription

拼接后解码结果: 
User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. 
我的早餐吃了大米，午餐吃了豆浆，晚餐吃了羊排。streaming_transcription
你的饮食挺丰富的哦！不过要注意营养均衡，尽量搭配一些蔬菜和水果。这样会让你的饮食更加健康哦！有什么我可以帮你的吗？

 <|audio_14295|> <|audio_7998|> <|audio_2360|> <|audio_8717|> <|audio_15616|> <|audio_2204|> <|audio_176|> <|audio_5783|> <|audio_3792|> <|audio_11346|> <|audio_4904|> <|audio_11123|> <|audio_10401|> <|audio_3960|> <|audio_15918|> <|audio_4849|> <|audio_16076|> <|audio_15503|> <|audio_2710|> <|audio_12072|> <|audio_12072|> <|audio_12072|> <|audio_12072|> <|audio_12072|> <|audio_5411|> <|audio_11795|> <|audio_6730|> <|audio_15029|> <|audio_13067|> <|audio_8288|> <|audio_3793|> <|audio_3246|> <|audio_9036|> <|audio_3161|> <|audio_8581|> <|audio_10654|> <|audio_13546|> <|audio_14636|> <|audio_2591|> <|audio_6191|> <|audio_6963|> <|audio_12072|> <|audio_12072|> <|audio_7661|> streaming_transcription
我爱你，学计算机。我爱你，学计算机。我爱你，学 <|audio_14295|> <|audio_15146|> <|audio_15870|> <|audio_14052|> <|audio_16238|> <|audio_8175|> <|audio_9551|> <|audio_1289|> <|audio_4176|> <|audio_260|> <|audio_6026|> <|audio_2664|> <|audio_6963|> <|audio_3407|> <|audio_2471|> <|audio_3192|> <|audio_10241|> <|audio_3254|> <|audio_4853|> <|audio_11327|> <|audio_4728|> <|audio_14258|> <|audio_11977|> <|audio_4794|> <|audio_5497|> <|audio_3232|> 计算机。我爱你，学计算机。我爱你，学计算机。 <|audio_12714|> <|audio_5492|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_2082|> <|audio_16007|> <|audio_14383|> <|audio_14799|> <|audio_8397|> <|audio_11747|> <|audio_6085|> <|audio_2476|> <|audio_13727|> <|audio_260|> <|audio_6026|> <|audio_3547|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_15312|> <|audio_5835|> <|audio_2414|> <|audio_6105|> <|audio_11396|> <|audio_5336|> <|audio_15659|> <|audio_8767|> <|audio_3580|> <|audio_8275|> <|audio_4794|> <|audio_5497|> <|audio_9640|> <|audio_14514|> <|audio_11143|> <|audio_7813|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_2082|> <|audio_16007|> <|audio_14383|> <|audio_7552|> <|audio_8397|> <|audio_11747|> <|audio_6085|> <|audio_2476|> <|audio_13727|> <|audio_260|> <|audio_6026|> <|audio_3547|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_6963|> <|audio_12476|> <|audio_13532|> <|audio_3192|> <|audio_10241|> <|audio_3254|> <|audio_1802|> <|audio_11327|> <|audio_4728|> <|audio_14258|> <|audio_7506|> <|audio_4794|> <|audio_5497|> <|audio_3232|> <|audio_10270|> <|audio_7813|> <|audio_7661|> <|audio_8173|> <|audio_7661|> <|audio_12072|> <|audio_12072|> <|audio_2066|> <|audio_14415|> <|audio_15146|> <|audio_3044|> <|audio_13178|> <|audio_14799|> <|audio_8397|> <|audio_11747|> <|audio_6085|> <|audio_2476|> <|audio_13727|> <|audio_260|> <|audio_6026|> <|audio_3547|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_6963|> <|audio_12476|> <|audio_13532|> <|audio_3192|> <|audio_10241|> <|audio_3254|> <|audio_1802|> <|audio_11327|> <|audio_4728|> <|audio_14258|> <|audio_7506|> <|audio_4794|> <|audio_5497|> <|audio_3232|> <|audio_10270|> <|audio_7813|> <|audio_7661|> <|audio_12072|> <|audio_12072|> <|audio_12072|> 
 <|audio_14295|> <|audio_7998|> <|audio_2360|> <|audio_8717|> <|audio_15616|> <|audio_2204|> <|audio_176|> <|audio_5783|> <|audio_3792|> <|audio_11346|> <|audio_4904|> <|audio_11123|> <|audio_10401|> <|audio_3960|> <|audio_15918|> <|audio_4849|> <|audio_16076|> <|audio_15503|> <|audio_2710|> <|audio_12072|> <|audio_12072|> <|audio_12072|> <|audio_12072|> <|audio_12072|> <|audio_5411|> <|audio_11795|> <|audio_6730|> <|audio_15029|> <|audio_13067|> <|audio_8288|> <|audio_3793|> <|audio_3246|> <|audio_9036|> <|audio_3161|> <|audio_8581|> <|audio_10654|> <|audio_13546|> <|audio_14636|> <|audio_2591|> <|audio_6191|> <|audio_6963|> <|audio_12072|> <|audio_12072|> <|audio_7661|> streaming_transcription
我爱你，学计算机。我爱你，学计算机。我爱你，学 <|audio_14295|> <|audio_15146|> <|audio_15870|> <|audio_14052|> <|audio_16238|> <|audio_8175|> <|audio_9551|> <|audio_1289|> <|audio_4176|> <|audio_260|> <|audio_6026|> <|audio_2664|> <|audio_6963|> <|audio_3407|> <|audio_2471|> <|audio_3192|> <|audio_10241|> <|audio_3254|> <|audio_4853|> <|audio_11327|> <|audio_4728|> <|audio_14258|> <|audio_7506|> <|audio_4794|> <|audio_5497|> <|audio_3232|> 计算机。我爱你，学计算机。我爱你，学计算机。 <|audio_15510|> <|audio_3039|> <|audio_7813|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_2082|> <|audio_16007|> <|audio_14383|> <|audio_7552|> <|audio_16244|> <|audio_11747|> <|audio_6085|> <|audio_2476|> <|audio_13727|> <|audio_260|> <|audio_6026|> <|audio_3547|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_6963|> <|audio_12476|> <|audio_11015|> <|audio_3793|> <|audio_12060|> <|audio_3254|> <|audio_4853|> <|audio_11327|> <|audio_4728|> <|audio_14258|> <|audio_7506|> <|audio_4794|> <|audio_5497|> <|audio_3232|> <|audio_10270|> <|audio_3039|> <|audio_7813|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_14415|> <|audio_2082|> <|audio_3493|> <|audio_6730|> <|audio_316|> <|audio_16244|> <|audio_11747|> <|audio_6085|> <|audio_2476|> <|audio_13727|> <|audio_260|> <|audio_6026|> <|audio_3547|> <|audio_12217|> <|audio_12217|> <|audio_12217|> <|audio_6963|> <|audio_14708|> <|audio_11015|> <|audio_7458|> <|audio_12060|> <|audio_2594|> <|audio_9921|> <|audio_3514|> <|audio_7449|> <|audio_1016|> <|audio_1357|> <|audio_4794|> <|audio_5497|> <|audio_3232|> <|audio_10270|> <|audio_7813|> <|audio_8173|> <|audio_7661|> <|audio_12072|> <|audio_12072|> 
 <|audio_14295|> <|audio_7998|> <|audio_2360|> <|audio_8717|> <|audio_15616|> <|audio_2204|> <|audio_176|> <|audio_5783|> <|audio_3792|> <|audio_11346|> <|audio_4904|> <|audio_11123|> <|audio_10401|> <|audio_3960|> <|audio_15918|> <|audio_4849|> <|audio_16076|> <|audio_15503|> <|audio_2710|> <|audio_12072|> <|audio_12072|> <|audio_12072|> <|audio_12072|> <|audio_12072|> <|audio_5411|> <|audio_11795|> <|audio_6730|> <|audio_15029|> <|audio_13067|> <|audio_8288|> <|audio_3793|> <|audio_3246|> <|audio_9036|> <|audio_3161|> <|audio_8581|> <|audio_10654|> <|audio_13546|> <|audio_14636|> <|audio_2591|> <|audio_6191|> <|audio_6963|> <|audio_12072|> <|audio_12072|> <|audio_7661|> streaming_transcription

'''
