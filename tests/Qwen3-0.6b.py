# Qwen1.5-MoE-dp2.py
# torchrun --nproc-per-node=2 Qwen1.5-MoE-dp2.py

from vllm import LLM, SamplingParams
import os
import faulthandler
faulthandler.enable()

os.environ["RANK"] = str(0)
os.environ["LOCAL_RANK"] = str(0)
os.environ["SIZE"] = str(1)

# 这个也是必需的
os.environ["MASTER_ADDR"] = "localhost"  # 在 MPI 模式下，这些通常会被忽略
os.environ["MASTER_PORT"] = "29500"  # 但设置它们是好习惯


# Use identical prompts to check consistency across DP ranks
prompts = [
    "欧盟有多少个国家，详细展开论述欧盟现状",
    "详细展开论述欧盟现状",
]

# Greedy decoding to eliminate randomness
sampling_params = SamplingParams(temperature=0.0)
sampling_params.max_tokens = 5

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    distributed_executor_backend="external_launcher",
    max_model_len=2048,
    seed=1,
    enforce_eager=True,
)

dp_rank = llm.llm_engine.vllm_config.parallel_config.data_parallel_rank

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(
        f"DP Rank: {dp_rank} Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n"
    )