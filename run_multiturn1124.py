

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
import pdb
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
def load_dialogues_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
def get_current_memory_gb():
    return round(torch.cuda.memory_allocated() / (1024 ** 3), 4)  # 保留四位小数

def get_max_memory_gb():
    return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4)

asr_model_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/huggingface.co/FunAudioLLM/SenseVoiceSmall"
asrmodel = ASRMODEL(
    model=asr_model_dir,
)
def dialog_id_exists(output_json, new_dialog_id):
    with open(output_json, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("dialog_id") == new_dialog_id:
                    return True
            except json.JSONDecodeError:
                continue
    return False

def encode_text(text, tokenizer):
    return tokenizer([text], return_tensors="pt")['input_ids'][0].tolist()

def encode_audio(audio_path, whisper_model, feature_extractor):
    return extract_speech_token(whisper_model, feature_extractor, [audio_path])[0]

def audio_tokens_to_str(audio_tokens):
    return "<|begin_of_audio|>" + "".join([f"<|audio_{x}|>" for x in audio_tokens]) + "<|end_of_audio|>"

def tokens_to_str(token_ids, tokenizer):
    return tokenizer.decode(token_ids, spaces_between_special_tokens=False)

def interleave_text_audio_tokens(text_token_ids, audio_token_ids, text_chunk=13, audio_chunk=26):
    result = []
    idx_text, idx_audio = 0, 0
    while idx_text < len(text_token_ids) or idx_audio < len(audio_token_ids):
        if idx_text < len(text_token_ids):
            result.extend(text_token_ids[idx_text:idx_text+text_chunk])
            idx_text += text_chunk
        if idx_audio < len(audio_token_ids):
            for tid in audio_token_ids[idx_audio:idx_audio+audio_chunk]:
                result.append(f"<|audio_{tid}|>")
            idx_audio += audio_chunk
    return result


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

    # print('all_inputs',len(all_inputs))
    start_time = time.time()
    all_inputs_emb = glm_tokenizer([all_inputs], return_tensors="pt").to(device)
    end_time = time.time()

    tokenizer_time = end_time - start_time

    
    all_input_ids = all_inputs_emb['input_ids']
    all_input_position_ids = all_inputs_emb['position_ids']
    all_input_attention_mask = all_inputs_emb['attention_mask']
    input_attention_mask = all_input_attention_mask
    # print('all_input_ids',all_input_ids.shape)
    # print('input_attention_mask',input_attention_mask.shape)
    # print('all_input_position_ids',all_input_position_ids.shape)
    


    print('all_input_ids',all_input_ids.shape[1])
    # if past_key_values is not None:
    #     print('past_key_values',past_key_values[0][0].shape[2])


    # print('previous_convert_token',previous_convert_token)


    # if past_key_values is not None:
    #     past_length = past_key_values[0][0].shape[2]
    #     print(past_length)
    #     input_ids = all_input_ids[:, past_length:]
        # if convert_token is not None:
        #     input_position_ids = all_input_position_ids[:,start+convert_ids.shape[1]+(all_input_ids.shape[1]-end):start+convert_ids.shape[1]+(all_input_ids.shape[1]-end)+all_input_ids.shape[1]]
        # else:
        #     input_position_ids = all_input_position_ids[:, past_length:]
        
        # print('all_input_position_ids_after',all_input_position_ids)

    start_time = time.time()


    if convert_token is not None and (mode == 0 or mode == 4 or mode ==5):
        # print('convert_token',convert_token)
        
        convert_ids = glm_tokenizer([convert_token], return_tensors="pt").to(device)['input_ids'][:,2:]
        # print('convert_ids',convert_ids)
        # print('convert_ids', glm_tokenizer([convert_token], return_tensors="pt").to(device)['input_ids'])
        previous_convert_ids = glm_tokenizer([previous_convert_token], return_tensors="pt").to(device)['input_ids'][:,2:]
        # print('previous_convert_ids',previous_convert_ids)

        def find_sublist_index(big_list, sub_list):
            """在big_list中查找sub_list的起止下标，返回(start, end)"""
            for i in range(len(big_list) - len(sub_list) + 1):
                if (big_list[i:i+len(sub_list)] == sub_list).all():
                    return i, i+len(sub_list)
            return None
        start,end = find_sublist_index(all_input_ids[0], previous_convert_ids[0])
        # print('start',start)
        # print('end',end)
        convert_position = all_input_position_ids[:, start:start+convert_ids.shape[1]]
        end_convert_ids = all_input_ids[:,end:]
        # print('end_convert_ids.shape[0]',end_convert_ids.shape[0])
        endconvert_position = all_input_position_ids[:, start+convert_ids.shape[1]:start+convert_ids.shape[1]+end_convert_ids.shape[1]]
        # print('111111111111111lengthbefore',all_input_ids.shape[1])
        
        # print('attention_mask',input_attention_mask.shape[1])

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

    streamer = TokenStreamer(skip_prompt=True)

    # if past_key_values is not None:
    #     print('aaa', type(past_key_values), len(past_key_values))
    #     print("第0层 ", past_key_values[0][0].shape)

    # print('input_ids',input_ids.shape)
    # print('attention_mask',inputs['attention_mask'].shape)
    # print('111convert_ids',convert_ids)
    # print('111end_convert_ids',end_convert_ids)
    # print('111convert_position',convert_position)
    # print('111endconvert_position',endconvert_position)
    # print('111delete_start',start)
    # print('111delete_end',end)


    if start is not None and (mode == 0 or mode == 4 or mode ==5):
        # print('333convert_ids', glm_tokenizer.decode(convert_ids.squeeze(0).tolist(), spaces_between_special_tokens=False))
        # print('3333end_convert_ids', glm_tokenizer.decode(end_convert_ids.squeeze(0).tolist(), spaces_between_special_tokens=False))
        # print('333input_ids',glm_tokenizer.decode(input_ids.squeeze(0).tolist(), spaces_between_special_tokens=False))
        # print('111111111111',start+convert_ids.shape[1])
        length = all_input_attention_mask.shape[1]-input_ids.shape[1]-end
        gen_kwargs = dict(  
            input_ids=convert_ids,
            position_ids=convert_position,
            attention_mask=all_input_attention_mask[:,:start+convert_ids.shape[1]],
            max_new_tokens=1,
            temperature=float(temperature),
            top_p=float(top_p),
            return_dict_in_generate=True,
            convert_position=all_input_position_ids[:,end:end+length],
            endconvert_position=all_input_position_ids[:,start+convert_ids.shape[1]:start+convert_ids.shape[1]+length],
            delete_start=start,
            delete_end=end,
            past_key_values=past_key_values,
            mode = mode,
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

     
    end_time = time.time()
    edit_time = end_time - start_time

    input_ids_length = input_ids.shape[-1] if hasattr(input_ids, "shape") else len(input_ids)
    if past_key_values is not None:
        kvcache_length = past_key_values[0][0].shape[2]
    else:
        kvcache_length = None

    
    print('input_ids',input_ids.shape)
    print('attention_mask',input_attention_mask.shape)
    print('position_ids',input_position_ids.shape)

    if mode == 5 and convert_token is not None:
        # 此时把kvcache置空
        all_inputs = all_inputs.replace(previous_convert_token, convert_token)

        all_inputs_emb = glm_tokenizer([all_inputs], return_tensors="pt").to(device)

        


        input_ids = all_inputs_emb['input_ids']
        input_position_ids = all_inputs_emb['position_ids']
        input_attention_mask = all_inputs_emb['attention_mask']
        print('bbbbbbbinput_ids',len(input_ids))
        print('input_attention_mask',len(input_attention_mask))
        


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
    # print()


    output_container = {}

    start_time = time.time()
    # pdb.set_trace()
    
    output = glm_model.generate(**gen_kwargs)


    if hasattr(output, "past_key_values"):
        print('kkkkk', output.past_key_values[0][0].shape)
        output_container["past_key_values"] = output.past_key_values
    elif isinstance(output, tuple) and len(output) > 1:
        output_container["past_key_values"] = output[1]

    for idx, token_id in enumerate(streamer):
        if idx == 0:
            first_token_time = time.time()
            first_token_time = first_token_time - start_time
            # pdb.set_trace()
        yield ("token", token_id)
    yield ("past_key_values", output_container.get("past_key_values", None))
    yield ("tokenizer_time", tokenizer_time)
    yield ("edit_time", edit_time)
    yield ("first_token_time", first_token_time)
    yield ("input_ids_length", input_ids_length)
    yield ("kvcache_length", kvcache_length)



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
        mode=0,
        repace_text=None
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

    if current_turn > save_history and (mode==0 or mode==4 or mode==5):
        convert_turn = current_turn - save_history - 1
        convert_history = extract_history_round(history,convert_turn)

        # 转译user text
        # res = asrmodel.generate(
        #         input=convert_history[0]['content']['path'],
        #         cache={},
        #         language="auto", 
        #         use_itn=True,
        #         batch_size_s=60,
        #         merge_vad=True,
        #         merge_length_s=15,
        #     )
        # convert_token = '<|user|>\n' + res[0]["text"].split('>')[-1] + '<|assistant|>streaming_transcription\n' + convert_history[2]['content']


        res = convert_history[0]['text']
        # print('res',res)
        convert_token = '<|user|>\n' + res + '<|assistant|>streaming_transcription\n' + convert_history[2]['content']


        previous_convert_token = extract_user_round(previous_input_tokens,convert_turn)

        # convert_token = previous_convert_token

    if input_mode == "audio":
        assert audio_path is not None
        history.append({"role": "user", "content": {"path": audio_path},"text":repace_text})
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
        history.append({"role": "user", "content": input_text})
        user_input = input_text
        system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
    
    system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."

    
    
    # tokenized = glm_tokenizer([inputs], return_tensors="pt")
    # input_ids = tokenized['input_ids'].to(device)


    
    all_inputs = previous_input_tokens + previous_completion_tokens
    if "<|system|>" not in all_inputs:
        all_inputs += f"<|system|>\n{system_prompt}"
    all_inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

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

    # 定义变量
    new_kv_cache = None
    tokenizer_time = None
    edit_time = None
    first_token_time = None
    memory = None
    input_ids_length = None
    kvcache_length = None


    complete_tokens, audio_tokens, text_tokens = [], [], []
    is_finalize = False

    if convert_token is not None and (mode == 0 or mode ==4 or mode ==5):
        all_inputs = all_inputs.replace(previous_convert_token, convert_token)


    for item in gen:
        key, value = item
        if key == "token":
            token_id = value
            # 统计首token显存
            if len(complete_tokens) == 0:
                memory = torch.cuda.memory_allocated() / (1024 ** 3)  # 单位GB
            if token_id == end_token_id:
                is_finalize = True
            if not is_finalize:
                complete_tokens.append(token_id)
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)
            if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                block_size = 20
                tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
                if prev_mel is not None and tts_mels:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                tts_speech, tts_mel = audio_decoder.token2wav(
                    tts_token, uuid=this_uuid,
                    prompt_token=flow_prompt_speech_token.to(device),
                    prompt_feat=prompt_speech_feat.to(device),
                    finalize=is_finalize
                )
                prev_mel = tts_mel
                tts_speechs.append(tts_speech.squeeze())
                tts_mels.append(tts_mel)
                flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                audio_tokens = []

        elif key == "past_key_values":
            new_kv_cache = value
        elif key == "tokenizer_time":
            tokenizer_time = value
        elif key == "edit_time":
            edit_time = value
        elif key == "first_token_time":
            first_token_time = value
        elif key == "input_ids_length":
            input_ids_length = value
        elif key == "kvcache_length":
            kvcache_length = value

    # 后续处理（音频合成、文本解码等）
    tts_speech = torch.cat(tts_speechs, dim=-1).cpu() if tts_speechs else None
    complete_text = glm_tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)
    audio_file = None
    if tts_speech is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_file = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice'+f.name
            torchaudio.save(audio_file, tts_speech.unsqueeze(0), 22050, format="wav")
        print(f"[Assistant Audio Output Saved]: {audio_file}")

    # history.append({"role": "assistant", "content": {"path": audio_file, "type": "audio/wav"}})
    # history.append({"role": "assistant", "content": glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)})

    model_output = glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)

    # 返回时多加三个时间
    return history, all_inputs, complete_text, audio_file, new_kv_cache, first_token_time, memory, tokenizer_time, edit_time,model_output,input_ids_length,kvcache_length


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--flow-path", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/huggingface.co/THUDM/glm-4-voice-decoder")
    parser.add_argument("--model-path", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/huggingface.co/THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", type= str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/huggingface.co/THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--max_duration", type=float, default=60.0, help="累计音频时长阈值")
    parser.add_argument("--k", type=int, default=1, help="累计音频时长阈值")



    args = parser.parse_args()
    device = "cuda"
    # 初始化
    print("Initializing models, please wait ...")

    initialize_fn(args)
    print("Initialization done.")
    initial_model_memory_gb = get_current_memory_gb()


    import glob

    txt_folder = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/data/LongDialogue/LongDialogue/audio_split_text"
    wav_base = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/data/LongDialogue/LongDialogue/audio_5700_test_split"

    txt_files = glob.glob(os.path.join(txt_folder, "*.txt"))


    def get_wav_duration(wav_path):
        info = torchaudio.info(wav_path)
        return info.num_frames / info.sample_rate


    json_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/data/LongDialogue/LongDialogue/LongDialogue_changed.json"
    with open(json_path, "r", encoding="utf-8") as f:
        dialogue_questions = json.load(f)

    results = []
    # for txt_file in txt_files[::-1]:
    for txt_file in txt_files:
        dialog_id = os.path.splitext(os.path.basename(txt_file))[0]  # e.g. MUL0001

        # output_json = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/tmp.json"
        output_json = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/mem1/multiturn_asr_mode{args.mode}_dur{args.max_duration}_k{args.k}.json"
        # output_json = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/tmp/multiturn_mode{args.mode}_dur{args.max_duration}.json"
       
        # 如果文件不存在，创建一个空的JSON文件
        if not os.path.exists(output_json):
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

        if dialog_id_exists(output_json, dialog_id):
            continue

        # 读取txt内容
        turns = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                wav_name, text = line.split(":", 1)
                wav_name = wav_name.strip()
                text = text.strip()
                wav_path = os.path.join(wav_base, dialog_id, wav_name)
                turns.append({"wav": wav_path, "text": text})

        # 累计时长
        # 逆序累计时长
        total_duration = 0
        end_idx = len(turns) - 1
        start_idx = end_idx
        max_duration = args.max_duration
        history = []
        input_tokens = ""
        completion_tokens = ""
        kv_cache = None

        while start_idx >= 0:
            wav_path = turns[start_idx]["wav"]
            duration = get_wav_duration(wav_path)

            wav_path1 = turns[start_idx-1]["wav"]
            duration1 = get_wav_duration(wav_path)

            total_duration += duration
            total_duration += duration1

            if total_duration > max_duration:
                break
            start_idx -= 2

        # start_idx+1 ~ end_idx 是选中的区间
        selected_idx = start_idx + 1

        if args.mode!=3:
            for i in range(selected_idx, end_idx + 1):
                if i % 2 == 1:
                    continue
                wav_path = turns[i]["wav"]
                text = turns[i]["text"]
                print('text',text)
                if not os.path.exists(wav_path):
                    print(f"Audio file does not exist: {wav_path}")
                    continue

                if args.mode==5:
                    tmp_mode = 0
                else:
                    tmp_mode = args.mode


                history, input_tokens, completion_tokens, audio_file, kv_cache, first_token_time, memory, tokenizer_time, edit_time ,_,_,_= inference_fn(
                    temperature=0.2,
                    top_p=0.8,
                    max_new_token=1,
                    input_mode='audio',
                    audio_path=wav_path,
                    input_text=None,
                    history=history,
                    previous_input_tokens=input_tokens,
                    previous_completion_tokens=completion_tokens,
                    previous_kv_cache=kv_cache,
                    save_history = args.k,
                    mode=tmp_mode,
                    repace_text = text,
                )
                turn = turns[i + 1]
                wav_path = turn["wav"]
                text = turn["text"]
                turnoutput = ""
                text_token_ids = encode_text(text, glm_tokenizer)
                audio_token_ids = encode_audio(wav_path, whisper_model, feature_extractor)
                interleaved = interleave_text_audio_tokens(text_token_ids, audio_token_ids, 13, 26)

                history.append({"role": "assistant", "content": {"path": wav_path, "type": "audio/wav"}})
                history.append({"role": "assistant", "content": text})

                for t in interleaved:
                    if isinstance(t, int):
                        turnoutput += glm_tokenizer.decode([t], spaces_between_special_tokens=False)
                    else:
                        turnoutput += t
                completion_tokens = turnoutput

            # 此时收集到了input_tokens和kv_cache
            collected_input_tokens = input_tokens
            collected_completion_tokens = completion_tokens
            collected_kv_cache = kv_cache
            collected_history = history

        else:
            for i in range(selected_idx, end_idx + 1):
                if i % 2 == 1:
                    continue
                wav_path = turns[i]["wav"]
                text = turns[i]["text"]


                history, input_tokens, completion_tokens, audio_file, kv_cache, first_token_time, memory, tokenizer_time, edit_time ,_,_,_= inference_fn(
                    temperature=0.2,
                    top_p=0.8,
                    max_new_token=1,
                    input_mode='text',
                    audio_path=None,
                    input_text=text,
                    history=history,
                    previous_input_tokens=input_tokens,
                    previous_completion_tokens=completion_tokens,
                    previous_kv_cache=kv_cache,
                    mode=args.mode,
                    repace_text = text

                )
                turn = turns[i + 1]
                wav_path = turn["wav"]
                text = turn["text"]
                turnoutput = ""
                text_token_ids = encode_text(text, glm_tokenizer)
                interleaved = text_token_ids

                history.append({"role": "assistant", "content": {"path": wav_path, "type": "audio/wav"}})
                history.append({"role": "assistant", "content": text})

                for t in interleaved:
                    if isinstance(t, int):
                        turnoutput += glm_tokenizer.decode([t], spaces_between_special_tokens=False)
                    else:
                        turnoutput += t
                completion_tokens = turnoutput

            # 此时收集到了input_tokens和kv_cache
            collected_input_tokens = input_tokens
            collected_completion_tokens = completion_tokens
            collected_kv_cache = kv_cache
            collected_history = history

        if args.mode==2 or args.mode==3 or args.mode==5:
            collected_kv_cache = None

        print(history)
        after_data_memory_gb = get_current_memory_gb()
        max_generate_memory_gb = initial_model_memory_gb


        for entry in dialogue_questions:
            if entry["sound_file"] != dialog_id:
                continue
            for qa in entry["conversations"]:
                mem_before = get_current_memory_gb()
                # if qa["HistoryDuration"] < max_duration:
                question_text = qa["Question"]
                turn = qa["Turn"]
                turn = int(turn)-1
                
                wav_path = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/data/LongDialogue/LongDialogue/questions/{dialog_id}_{turn}.wav"
                print('tttttttttttt',collected_input_tokens)
                # 推理
                history, _, complete_text, _, _, first_token_time, first_token_memory, tokenizer_time, edit_time, model_output,input_length,kvcache_length = inference_fn(
                    temperature=0.2,
                    top_p=0.8,
                    max_new_token=200,
                    input_mode='audio',
                    audio_path=wav_path,
                    input_text=None,
                    history=collected_history.copy(),
                    previous_input_tokens=collected_input_tokens,
                    previous_completion_tokens=collected_completion_tokens,
                    save_history = args.k,
                    previous_kv_cache=collected_kv_cache,
                    mode=args.mode
                )
                mem_after = get_current_memory_gb()
                max_generate_memory_gb = max(max_generate_memory_gb, mem_before, mem_after, get_max_memory_gb())
                
                # results.append({
                #     "dialog_id": dialog_id,
                #     "question": question_text,
                #     "answer": qa["Answer"],
                #     "model_output": model_output,
                #     "first_token_time": first_token_time,
                #     "first_token_memory": first_token_memory,
                #     "tokenizer_time": tokenizer_time,
                #     "edit_time": edit_time,
                #     "HistoryDuration": qa["HistoryDuration"]
                # })

                with open(output_json, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "dialog_id": dialog_id,
                        "question": question_text,
                        "answer": qa["Answer"],
                        "model_output": model_output,
                        "first_token_time": first_token_time,
                        "first_token_memory": first_token_memory,
                        "tokenizer_time": tokenizer_time,
                        "edit_time": edit_time,
                        "HistoryDuration": qa["HistoryDuration"],
                        "input_length":input_length,
                        "kvcache_length":kvcache_length,
                        "initial_model_memory_gb": initial_model_memory_gb,
                        "after_data_memory_gb": after_data_memory_gb,
                        "max_generate_memory_gb": max_generate_memory_gb
                    }, ensure_ascii=False) + "\n")

    # with open(f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/multiturn_mode{args.mode}_dur{max_duration}_2.json", "w", encoding="utf-8") as f:
        # json.dump(results, f, ensure_ascii=False, indent=2)





'''
conda activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/tools/env_bk/glmv
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice



export CUDA_VISIBLE_DEVICES=0
nohup python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/run_multiturn.py --mode 1 --max_duration 960 --k 1\
    > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/log/0.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/run_multiturn.py --mode 3 --max_duration 960 --k 1\
    > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/log/1.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/run_multiturn.py --mode 0 --max_duration 960 --k 1\
    > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/log/2.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/run_multiturn.py --mode 0 --max_duration 960 --k 3\
    > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/log/3.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/run_multiturn.py --mode 0 --max_duration 960 --k 5\
    > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/log/4.log 2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/run_multiturn.py --mode 4 --max_duration 960 --k 1\
    > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/log/5.log 2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/run_multiturn.py --mode 5 --max_duration 960 --k 1\
    > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/log/6.log 2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/run_multiturn.py --mode 5 --max_duration 960 --k 3\
    > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/log/7.log 2>&1 &






unzip /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/data/LongDialogue/questions.zip -d /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/data/LongDialogue/LongDialogue/question

mode = 0:kv编辑
mode = 1：有kv
mode = 2：原始的glm
mode = 3：前n轮是文本
mode = 4：直接替换
mode = 5：重新计算





'''
