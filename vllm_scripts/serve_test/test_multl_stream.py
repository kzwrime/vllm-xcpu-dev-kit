# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 使用说明:
#   方式1: 通过 -e 参数指定预设文件
#     python serve_test/test_multl_stream.py -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh
#
#   方式2: 通过 PRESET 环境变量
#     PRESET=serial/Qwen3-30B-A3B_dp1_tp1_eager python serve_test/test_multl_stream.py
#
#   方式3: 使用 user_env.sh
#     python serve_test/test_multl_stream.py
#
# 功能说明:
#   并发测试多个流式请求，每个请求的结果写入独立的日志文件 (vllm_task_X.log)
#   屏幕上显示实时状态和进度

import argparse
import asyncio
from collections import deque
import os
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any
import urllib.error
import urllib.request

# 移除 logging，改为直接写文件和动态终端输出
from openai import AsyncOpenAI


def get_stream_value(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is not None:
        return value
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name)
    return None


def describe_empty_output(
    *,
    content_chars: int,
    reasoning_chars: int,
    chunk_count: int,
    choice_chunk_count: int,
    finish_reasons: list[str],
    stop_reasons: list[str],
) -> str:
    if content_chars > 0:
        return ""

    finish_reason = finish_reasons[-1] if finish_reasons else "unknown"
    stop_reason = stop_reasons[-1] if stop_reasons else ""
    total_chars = content_chars + reasoning_chars

    details = [
        f"finish_reason={finish_reason}",
        f"stop_reason={stop_reason or 'None'}",
        f"stream_chunks={chunk_count}",
        f"choice_chunks={choice_chunk_count}",
        f"reasoning_chars={reasoning_chars}",
        f"content_chars={content_chars}",
    ]

    if chunk_count == 0:
        cause = "服务端返回了 HTTP 流，但客户端没有收到任何 chunk。"
    elif choice_chunk_count == 0:
        cause = "客户端收到了 chunk，但没有收到 choices；可能是服务端流式协议异常。"
    elif total_chars == 0 and finish_reason == "length":
        cause = (
            "生成达到 max_tokens 限制，但没有返回 reasoning/content 文本；"
            "可能生成的是被 reasoning parser 过滤的特殊 token、空白 token，"
            "或当前脚本还未识别该模型使用的输出字段。"
        )
    elif total_chars == 0 and finish_reason == "stop":
        cause = (
            "模型立即生成了 EOS/stop token，没有产生可见文本；"
            "通常和 chat template、stop 条件、采样结果或模型状态有关。"
        )
    elif reasoning_chars > 0 and finish_reason == "length":
        cause = (
            "max_tokens 在 reasoning 阶段已耗尽，因此还没进入 content 正文。"
        )
    elif reasoning_chars > 0 and finish_reason == "stop":
        cause = (
            "模型只返回了 reasoning，然后遇到 EOS/stop，未产生 content 正文。"
        )
    elif finish_reason == "content_filter":
        cause = "服务端报告 content_filter，输出被过滤。"
    elif finish_reason == "tool_calls":
        cause = "模型选择了 tool_calls 路径，没有产生 content 正文。"
    else:
        cause = (
            "请求成功但 content 为空；需要结合 finish_reason、stop_reason "
            "和原始 chunk 判断。"
        )

    return cause + "\n" + "\n".join(f"- {detail}" for detail in details)


def chunk_to_json(chunk: Any) -> str:
    dump_json = getattr(chunk, "model_dump_json", None)
    if callable(dump_json):
        return dump_json(exclude_none=False)
    return repr(chunk)


def format_exception_summary(exc: Exception) -> str:
    message = str(exc) or repr(exc)
    parts = [f"{type(exc).__name__}: {message}"]
    cause = getattr(exc, "__cause__", None)
    context = getattr(exc, "__context__", None)
    if cause is not None:
        parts.append(f"cause={type(cause).__name__}: {cause}")
    if context is not None and context is not cause:
        parts.append(f"context={type(context).__name__}: {context}")
    return " | ".join(parts)


def format_exception_details(exc: Exception) -> str:
    lines = [format_exception_summary(exc)]
    for attr in ["status_code", "code", "type", "param", "request_id", "body"]:
        value = getattr(exc, attr, None)
        if value is not None:
            lines.append(f"{attr}: {value!r}")

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        text = getattr(response, "text", None)
        if status_code is not None:
            lines.append(f"response.status_code: {status_code!r}")
        if text:
            lines.append(f"response.text: {text[:2000]!r}")

    lines.append("")
    lines.append("[Traceback]")
    lines.extend(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return "\n".join(lines).rstrip()


def check_server_ready(port: str) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
    except (urllib.error.URLError, OSError, RuntimeError) as exc:
        print(f"[错误] vLLM API 不可访问: {url}")
        print(f"       异常: {format_exception_summary(exc)}")
        print("       请先启动服务，并确认以下命令可访问:")
        print(f"       curl {url}")
        sys.exit(1)


def source_env_with_preset(script_dir, preset_file=None):
    """
    使用与 serve_test_template.sh 相同的逻辑加载环境变量

    支持两种方式:
    1. -e 参数指定预设文件路径 (如: ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh)
    2. PRESET 环境变量 (如: serial/Qwen3-30B-A3B_dp1_tp1_eager)

    加载优先级:
    1. -e 参数指定的预设文件
    2. PRESET 环境变量
    3. 用户自定义配置 (user_env.sh)
    4. 模板文件回退 (user_env_template.sh)
    """
    vllm_scripts_dir = os.path.abspath(os.path.join(script_dir, ".."))
    common_sh = os.path.join(vllm_scripts_dir, "common.sh")

    # 构建与 parse_args_and_load_env 相同的 bash 命令
    bash_command = f"""
set -e
source {shlex.quote(common_sh)}
load_env_file {shlex.quote(os.path.join(vllm_scripts_dir, "env.sh"))}
"""

    if preset_file:
        bash_command += f"""
load_preset_file {shlex.quote(preset_file)}
"""
    else:
        bash_command += f"""
load_user_config {shlex.quote(vllm_scripts_dir)}
"""

    bash_command += f"""
apply_runtime_overrides
# 输出所有环境变量
env
"""

    result = subprocess.run(
        ["bash", "-c", bash_command],
        capture_output=True,
        text=True,
        env=os.environ,  # 传递当前进程的环境变量（包括 PRESET）
    )

    if result.returncode != 0:
        print("[错误] 加载环境变量失败:")
        print(result.stderr)
        sys.exit(1)

    env_vars = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value
    return env_vars


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="并发测试 vLLM 流式输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 通过 -e 参数指定预设文件
  python test_multl_stream.py -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh

  # 限制每个请求的回答长度
  python test_multl_stream.py -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh --max-tokens 16

  # 设置采样温度
  python test_multl_stream.py -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh --temperature 0

  # 显式开启 thinking（默认关闭）
  python test_multl_stream.py -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh --enable-thinking

  # 通过 PRESET 环境变量
  PRESET=serial/Qwen3-30B-A3B_dp1_tp1_eager python test_multl_stream.py

  # 使用 user_env.sh
  python test_multl_stream.py
        """
    )
    parser.add_argument(
        "-e",
        metavar="预设文件",
        help="指定预设文件路径（相对于当前目录或绝对路径）"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="限制每个请求的最大输出 token 数，默认 32",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度，默认 0.7",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="开启模型 thinking",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="关闭模型 thinking（默认）",
    )
    parser.set_defaults(enable_thinking=False)
    return parser.parse_args()


# 1. 解析命令行参数
args = parse_args()

# 2. 使用与 serve_test_template.sh 相同的逻辑加载环境变量
script_dir = os.path.dirname(os.path.abspath(__file__))
env_vars = source_env_with_preset(script_dir, preset_file=args.e)

# 3. 从环境变量中获取配置
MODEL_NAME = env_vars.get("USER_VLLM_MODEL", "你的模型名称")
PORT = env_vars.get("USER_VLLM_PORT", "8000")
MAX_TOKENS = args.max_tokens
if MAX_TOKENS <= 0:
    print("[错误] --max-tokens 必须大于 0")
    sys.exit(1)
TEMPERATURE = args.temperature
if TEMPERATURE < 0:
    print("[错误] --temperature 必须大于等于 0")
    sys.exit(1)
ENABLE_THINKING = args.enable_thinking

# 4. 初始化异步客户端
client = AsyncOpenAI(
    api_key="EMPTY",  # vLLM 默认不需要真实的 API Key
    base_url=f"http://localhost:{PORT}/v1",
)

# 2. 准备多组测试数据
PROMPTS = [
    "请用一段话简单介绍一下量子计算。",
    "写一首关于春天的七言绝句。",
    "欧盟有多少个国家，详细展开论述欧盟现状。",
    "解释一下相对论的核心思想。",
    "给出三个提高编程效率的建议。",
    (
        "作为一名资深的国际政治与经济评论员，"
        "请针对欧洲联盟（EU）的现状进行深度剖析。"
        "首先，"
        "请明确指出截至2026年欧盟的成员国数量，"
        "并简述近年来成员国变动（如英国脱欧后）对联盟地缘政治版图的实质性影响。"
        "接下来，"
        "请从以下三个维度详细展开论述欧盟的现状：\n\n1. **经济韧性与数字化转型**："
        "在面对全球通胀压力及能源危机后，"
        "欧盟目前的单一市场表现如何？其《数字市场法案》（DMA）和《数字服务法案》（DSA）"
        "在实施过程中对成员国经济活力产生了怎样的正面或负面效应？\n\n2. "
        "**政治一体化与内部博弈**：请探讨‘多速欧洲’概念在当前的实践情况。"
        "特别是针对匈牙利、波兰等国与欧盟总部在法治原则上的博弈，"
        "以及这种内部撕裂是否正在削弱欧盟作为一个整体在国际事务中的话语权。"
        "\n\n3. **外部安全与防务自主**：在当前的国际安全形势下，"
        "欧盟是如何平衡其对北约的依赖与对‘战略自主’（Strategic Autonomy）的追求的？"
        "欧盟防务共同体的建设目前处于什么阶段？\n\n最后，"
        "请对欧盟未来五年的发展趋势做一个简短但具前瞻性的预测，"
        "分析其是否仍能维持全球第三大经济体的核心地位。"
        "请保持语言风格专业、客观且逻辑严密。"
    ),
]
if env_vars.get("VLLM_PD_MULTI_INCLUDE_LONG") != "1":
    PROMPTS = PROMPTS[:-1]  # 最后一条过长的测试用例暂时不执行

# 用于记录各个任务的状态，方便在屏幕上更新 tips
task_states = {}


async def fetch_stream(task_id: int, prompt: str):
    """处理单个流式请求，并将结果实时写入独立的日志文件"""
    task_states[task_id] = {
        "status": "请求中",
        "reasoning_chars": 0,
        "content_chars": 0,
        "empty_reason": "",
        "error": "",
        "log_file": os.path.abspath(f"vllm_task_{task_id}.log"),
    }
    log_file = f"vllm_task_{task_id}.log"

    try:
        # 打开独立的日志文件
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"【Prompt】: {prompt}\n")
            f.write("=" * 40 + "\n")

            # 发起异步流式请求
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=True,  # 开启流式输出
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": ENABLE_THINKING,
                    }
                },
            )

            reasoning_text = ""
            output_text = ""
            wrote_reasoning_header = False
            wrote_content_header = False
            chunk_count = 0
            choice_chunk_count = 0
            reasoning_chunk_count = 0
            content_chunk_count = 0
            finish_reasons = []
            stop_reasons = []
            last_chunks = deque(maxlen=3)

            # 实时接收数据块，显式区分思考过程和最终回答
            async for chunk in response:
                chunk_count += 1
                last_chunks.append(chunk_to_json(chunk))
                choices = get_stream_value(chunk, "choices") or []
                if not choices:
                    continue
                choice_chunk_count += 1

                choice = choices[0]
                finish_reason = get_stream_value(choice, "finish_reason")
                if finish_reason:
                    finish_reasons.append(str(finish_reason))
                stop_reason = get_stream_value(choice, "stop_reason")
                if stop_reason is not None:
                    stop_reasons.append(str(stop_reason))

                delta = get_stream_value(choice, "delta")
                reasoning = (
                    get_stream_value(delta, "reasoning_content")
                    or get_stream_value(delta, "reasoning")
                    or ""
                )
                content = get_stream_value(delta, "content") or ""

                if reasoning:
                    reasoning_chunk_count += 1
                    reasoning_text += reasoning
                    if not wrote_reasoning_header:
                        f.write("[reasoning_content]\n")
                        wrote_reasoning_header = True
                    f.write(reasoning)
                    f.flush()  # 流式：立刻将内容写入磁盘
                    task_states[task_id]["reasoning_chars"] = len(reasoning_text)
                    task_states[task_id]["status"] = "思考中"

                if content:
                    content_chunk_count += 1
                    output_text += content
                    if not wrote_content_header:
                        if wrote_reasoning_header:
                            f.write("\n\n[content]\n")
                        else:
                            f.write("[content]\n")
                        wrote_content_header = True
                    f.write(content)
                    f.flush()  # 流式：立刻将内容写入磁盘
                    task_states[task_id]["content_chars"] = len(output_text)
                    task_states[task_id]["status"] = "生成中"

            empty_reason = describe_empty_output(
                content_chars=len(output_text),
                reasoning_chars=len(reasoning_text),
                chunk_count=chunk_count,
                choice_chunk_count=choice_chunk_count,
                finish_reasons=finish_reasons,
                stop_reasons=stop_reasons,
            )
            if len(output_text) > 0:
                task_states[task_id]["status"] = "已完成"
            elif len(reasoning_text) > 0:
                task_states[task_id]["status"] = "空正文"
            else:
                task_states[task_id]["status"] = "空输出"
            task_states[task_id]["empty_reason"] = empty_reason
            f.write(
                "\n\n"
                + "=" * 40
                + "\n"
                + f"[统计] reasoning_chars={len(reasoning_text)}, "
                + f"content_chars={len(output_text)}, "
                + f"stream_chunks={chunk_count}, "
                + f"choice_chunks={choice_chunk_count}, "
                + f"reasoning_chunks={reasoning_chunk_count}, "
                + f"content_chunks={content_chunk_count}, "
                + f"finish_reasons={finish_reasons}, "
                + f"stop_reasons={stop_reasons}\n"
            )
            if empty_reason:
                f.write("[空输出诊断]\n")
                f.write(empty_reason + "\n")
                f.write("[最近原始 chunk]\n")
                for idx, raw_chunk in enumerate(last_chunks, 1):
                    f.write(f"chunk[-{len(last_chunks) - idx + 1}]: {raw_chunk}\n")
            f.write("【生成结束】")

    except Exception as e:
        task_states[task_id]["status"] = "错误"
        task_states[task_id]["error"] = format_exception_summary(e)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n\n[请求失败]\n")
            f.write(format_exception_details(e))
            f.write("\n")


async def display_tips():
    """在屏幕上维持动态的提示信息 (Tips)"""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    start_time = time.time()

    while True:
        active_count = 0
        status_strs = []

        for i in range(len(PROMPTS)):
            state = task_states.get(
                i,
                {
                    "status": "等待中",
                    "reasoning_chars": 0,
                    "content_chars": 0,
                    "empty_reason": "",
                },
            )
            if state["status"] not in [
                "已完成",
                "错误",
                "等待中",
                "空正文",
                "空输出",
            ]:
                active_count += 1
            reasoning_chars = state.get("reasoning_chars", 0)
            content_chars = state.get("content_chars", 0)
            status_strs.append(
                f"T{i}: {state['status']}"
                f"(思考{reasoning_chars}字符/正文{content_chars}字符)"
            )

        elapsed = time.time() - start_time
        spinner = frames[idx % len(frames)]

        # 拼接单行状态栏 (\033[K 用于清除行尾残余字符，防止残留)
        line = (
            f"\r\033[K{spinner} 耗时: {elapsed:.1f}s"
            + f" | 活跃任务: {active_count}/{len(PROMPTS)} | "
            + " | ".join(status_strs)
        )

        sys.stdout.write(line)
        sys.stdout.flush()

        # 如果所有任务都已完成或出错，则退出循环
        if active_count == 0 and all(
            task_states.get(i, {}).get("status") in [
                "已完成",
                "错误",
                "空正文",
                "空输出",
            ]
            for i in range(len(PROMPTS))
        ):
            sys.stdout.write(
                "\n\n所有并发任务执行完毕！请查看各自的 vllm_task_X.log 文件。\n"
            )
            break

        idx += 1
        await asyncio.sleep(0.15)


async def main():
    print("开始并发测试流式输出")
    print(f"模型: {MODEL_NAME}")
    print(f"端口: {PORT}")
    print(f"max_tokens: {MAX_TOKENS} (reasoning + content 总生成 token 限制)")
    print(f"temperature: {TEMPERATURE}")
    print(f"thinking: {'enabled' if ENABLE_THINKING else 'disabled'}")
    print(f"日志文件: {os.getcwd()}/vllm_task_*.log")
    print("提示：模型输出将流式写入独立的日志文件，屏幕仅显示实时状态。\n")
    check_server_ready(PORT)

    # 启动状态监控任务
    monitor_task = asyncio.create_task(display_tips())

    # 并发执行所有流式请求任务
    fetch_tasks = [fetch_stream(i, prompt) for i, prompt in enumerate(PROMPTS)]
    await asyncio.gather(*fetch_tasks)

    # 等待监控任务结束
    await monitor_task

    failed_tasks = [
        task_id for task_id, state in task_states.items()
        if state.get("status") == "错误"
    ]
    if failed_tasks:
        print("[错误] 以下任务失败:")
        for task_id in failed_tasks:
            state = task_states.get(task_id, {})
            error = state.get("error") or "未知错误"
            log_file = state.get("log_file") or os.path.abspath(
                f"vllm_task_{task_id}.log"
            )
            print(f"  - T{task_id}: {error}")
            print(f"    日志: {log_file}")
        sys.exit(1)

    empty_content_tasks = [
        task_id for task_id, state in task_states.items()
        if state.get("status") in ["空正文", "空输出"]
    ]
    if empty_content_tasks:
        print(f"[警告] 以下任务没有生成正文: {empty_content_tasks}")
        print("       具体原因请查看对应 vllm_task_X.log 的 [空输出诊断]。")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
