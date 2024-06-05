from peft import get_peft_config, get_peft_model, LoraConfig, PrefixTuningConfig, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
from transformers import BitsAndBytesConfig,LlamaTokenizer,LlamaForCausalLM,GPT2Tokenizer, GPT2LMHeadModel,AutoTokenizer 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from peft import prepare_model_for_kbit_training


quantization_8model_names = ['WizardCoder-7B','Guanaco-2-7B']

quantization_4model_names = ['LLaMA-2-7B-Chat','LLaMA-3-7B-Instruct','Alpaca-7B',
                            'WizardLM-2-7B','WizardLM-7B','WizardMath-7B']


def get_model(model_name,device):
    print(model_name)
    nf4_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_use_double_quant=True,
           bnb_4bit_compute_dtype=torch.bfloat16
        )
    if model_name == 'LLaMA-2-7B-Chat':
        model_path ="meta-llama/Llama-2-7b-chat-hf"
    elif model_name == 'LLaMA-3-7B-Instruct':
        model_path = model_path ="meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == 'Alpaca-7B':
        model_path="wxjiao/alpaca-7b"
    
    elif model_name == 'WizardLM-2-7B':
        model_path="dreamgen/WizardLM-2-7B"
    elif model_name == 'WizardLM-7B':
        print("라마라고?")
        model_path="TheBloke/wizardLM-7B-HF"
    elif model_name == 'WizardCoder-7B':
        model_path ="rombodawg/WizardCoder-Python-7B-V1.0_Sharded_1.5gb"
    elif model_name == 'WizardMath-7B':
        model_path ="WizardLMTeam/WizardMath-7B-V1.1"
    elif model_name == 'ChatGLM-6B':
        model_path ="THUDM/chatglm-6b"
    elif model_name == 'Guanaco-2-7B':
        model_path ="KBlueLeaf/guanaco-7b-leh-v2"
    
    if model_name in quantization_8model_names:
        model =AutoModelForCausalLM.from_pretrained(model_path,
                                                #quantization_config =nf4_config, #  4.4GB로 
                                                load_in_8bit=True,
                                                    device_map="auto", # gpu 꽉차면 cpu로 올려줌 
                                               cache_dir="/data/wooseok/huggingface/hub",
                                                use_auth_token = "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")
        if model_name: 
            tokenizer = LlamaTokenizer.from_pretrained(model_path,
                                            use_auth_token = "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")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                    use_auth_token = "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")
        
        tokenizer.pad_token = tokenizer.eos_token
    
    elif model_name in quantization_4model_names:
        model =AutoModelForCausalLM.from_pretrained(model_path,
                                                quantization_config =nf4_config, #  4.4GB로 
                                                #load_in_8bit=True,
                                                    device_map="auto", # gpu 꽉차면 cpu로 올려줌 
                                               cache_dir="/data/wooseok/huggingface/hub",
                                                use_auth_token = "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")
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                use_auth_token = "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")
        
        tokenizer.pad_token = tokenizer.eos_token
    
    elif model_name == 'ChatGLM-6B':
        model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True,
                                      #quantization_config =nf4_config,
                                      #load_in_8bit = True,
                                     device_map ='auto',
                                      cache_dir="/data/wooseok/huggingface/hub")
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b")
    
    else:
        if model_name =='DialoGPT-large':
            model_path = f"microsoft/{model_name}"
        elif model_name == 'GPT-2-large':
            model_path ='gpt2-large'
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_path,
                                                )
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_path,
                                               cache_dir="/data/wooseok/huggingface/hub",
                                               ).to(device)
    return tokenizer, model

