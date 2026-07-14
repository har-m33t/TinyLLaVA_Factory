from transformers import LlamaForCausalLM, AutoTokenizer

from . import register_llm


@register_llm('vicuna')
def return_vicunaclass():
    # Vicuna-7B (lmsys/vicuna-7b-v1.5) is a LlamaForCausalLM. The LLMFactory
    # substring-matches the checkpoint path, so any path containing "vicuna"
    # (e.g. lmsys/vicuna-7b-v1.5) selects this loader. Pair with the `llama`
    # conversation template (tinyllava/data/template/llama_template.py), which
    # is the Vicuna v1 format.
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
