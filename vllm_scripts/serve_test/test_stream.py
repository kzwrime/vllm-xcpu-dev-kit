# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from collections import deque
import os
import shlex
import subprocess
import sys
import traceback
from typing import Any
import urllib.error
import urllib.request

from openai import OpenAI


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
        cause = "max_tokens 在 reasoning 阶段已耗尽，因此还没进入 content 正文。"
    elif reasoning_chars > 0 and finish_reason == "stop":
        cause = "模型只返回了 reasoning，然后遇到 EOS/stop，未产生 content 正文。"
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


def main():
    parser = argparse.ArgumentParser(
        description="测试 vLLM 流式输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 通过 -e 参数指定预设文件
  python test_stream.py -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh

  # 设置输出长度和采样温度
  python test_stream.py -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh --max-tokens 3000 --temperature 0

  # 显式关闭 thinking（默认开启）
  python test_stream.py -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh --disable-thinking

  # 通过 PRESET 环境变量
  PRESET=serial/Qwen3-30B-A3B_dp1_tp1_eager python test_stream.py
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
        default=3000,
        help="限制请求的最大输出 token 数，默认 3000",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="采样温度，默认 0.5",
    )
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="跳过启动前 /v1/models 连通性检查",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="开启模型 thinking（默认）",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="关闭模型 thinking",
    )
    parser.set_defaults(enable_thinking=True)

    args = parser.parse_args()
    if args.max_tokens <= 0:
        print("[错误] --max-tokens 必须大于 0")
        sys.exit(1)
    if args.temperature < 0:
        print("[错误] --temperature 必须大于等于 0")
        sys.exit(1)

    # 1. 使用与 serve_test_template.sh 相同的逻辑加载环境变量
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_vars = source_env_with_preset(script_dir, preset_file=args.e)

    # 2. 从环境变量中获取配置
    MODEL_NAME = env_vars.get("USER_VLLM_MODEL", "你的模型名称")
    PORT = env_vars.get("USER_VLLM_PORT", "8000")
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature

    # 3. 初始化客户端，指向你的 vLLM 服务地址
    client = OpenAI(
        api_key="EMPTY",  # vLLM 默认不需要真实的 API Key
        base_url=f"http://localhost:{PORT}/v1",
    )

    print("开始测试流式输出...\n")
    print(f"模型: {MODEL_NAME}")
    print(f"端口: {PORT}")
    print(f"max_tokens: {MAX_TOKENS} (reasoning + content 总生成 token 限制)")
    print(f"temperature: {TEMPERATURE}")
    print(f"thinking: {'enabled' if args.enable_thinking else 'disabled'}")
    print("-" * 50)
    if not args.no_health_check:
        check_server_ready(PORT)

    try:
        # 4. 发起流式请求
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "量子计算是一种基于量子力学原理的全新信息处理模式，它利用量子比特（qubit）代替传统计算机使用的二进制比特，并通过量子叠加态和量子纠缠等现象实现远超经典计算机的>计算潜力。我希望你能用一段话简要介绍量子计算的基本概念，包括它的核心原理、关键技术特征（如叠加态、纠缠、干涉等）、目前的主要实现方式（例如超导量子比特、离子阱、光子量子计算等），以及它在科学研>究和产业应用中的潜在价值。此外，请结合当前量子计算的发展现状，谈谈主要技术挑战（如量子退相干、误差校正、规模化等）和未来可能的突破方向。最后，如果可以的话，请简要说明量子计算与经典计算的区别，>并举例说明在某些领域（如密码学、药物研发、人工智能、金融建模等）中，它可能带来的变革性影响。这样的一段话，既要通俗易懂，又要涵盖技术与应用的要点，以便读者在短时间内建立起对量子计算的整体认知。"},
            ],
            stream=True,  # 开启流式输出
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": args.enable_thinking,
                }
            },
        )

        # 5. 实时打印返回的数据块 (chunks)，显式区分思考过程和最终回答
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
        for chunk in response:
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
                    print("[reasoning_content]")
                    wrote_reasoning_header = True
                print(reasoning, end="", flush=True)

            if content:
                content_chunk_count += 1
                output_text += content
                if wrote_reasoning_header and not wrote_content_header:
                    print("\n\n[content]")
                    wrote_content_header = True
                # 使用 end="" 和 flush=True 确保文字能够逐字平滑显示
                print(content, end="", flush=True)

        empty_reason = describe_empty_output(
            content_chars=len(output_text),
            reasoning_chars=len(reasoning_text),
            chunk_count=chunk_count,
            choice_chunk_count=choice_chunk_count,
            finish_reasons=finish_reasons,
            stop_reasons=stop_reasons,
        )
        print(
            f"\n\n[统计] reasoning_chars={len(reasoning_text)}, "
            f"content_chars={len(output_text)}, "
            f"stream_chunks={chunk_count}, "
            f"choice_chunks={choice_chunk_count}, "
            f"reasoning_chunks={reasoning_chunk_count}, "
            f"content_chunks={content_chunk_count}, "
            f"finish_reasons={finish_reasons}, "
            f"stop_reasons={stop_reasons}"
        )
        if empty_reason:
            print("\n[空输出诊断]")
            print(empty_reason)
            print("[最近原始 chunk]")
            for idx, raw_chunk in enumerate(last_chunks, 1):
                print(f"chunk[-{len(last_chunks) - idx + 1}]: {raw_chunk}")

        # response = client.completions.create(
        #     model=MODEL_NAME,  # 注意：必须使用支持补全接口的模型
        #     prompt="请用一段话简单介绍一下量子计算。", # 这里是纯字符串，不是 messages 列表
        #     stream=True,
        #     temperature=0.5,
        #     max_tokens=3000,
        # )
        # for chunk in response:
        #     if chunk.choices[0].text:
        #         print(chunk.choices[0].text, end="", flush=True)

    except Exception as e:
        print("\n[请求失败]")
        print(format_exception_details(e))
        sys.exit(1)

    print("\n" + "-" * 50)
    print("生成结束！")


if __name__ == "__main__":
    main()
