# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import os
import subprocess
import sys

from openai import OpenAI


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
    # 构建与 parse_args_and_load_env 相同的 bash 命令
    bash_command = f"""
set -e
source {script_dir}/../common.sh

# 如果通过 -e 参数指定了预设文件
"""

    if preset_file:
        bash_command += f"""
load_preset_file "{preset_file}"
"""

    bash_command += f"""
# 否则使用常规的 load_user_config 逻辑
load_user_config "{script_dir}/.."

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

  # 通过 PRESET 环境变量
  PRESET=serial/Qwen3-30B-A3B_dp1_tp1_eager python test_stream.py
        """
    )
    parser.add_argument(
        "-e",
        metavar="预设文件",
        help="指定预设文件路径（相对于当前目录或绝对路径）"
    )

    args = parser.parse_args()

    # 1. 使用与 serve_test_template.sh 相同的逻辑加载环境变量
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_vars = source_env_with_preset(script_dir, preset_file=args.e)

    # 2. 从环境变量中获取配置
    MODEL_NAME = env_vars.get("USER_VLLM_MODEL", "你的模型名称")
    PORT = env_vars.get("USER_VLLM_PORT", "8000")

    # 3. 初始化客户端，指向你的 vLLM 服务地址
    client = OpenAI(
        api_key="EMPTY",  # vLLM 默认不需要真实的 API Key
        base_url=f"http://localhost:{PORT}/v1",
    )

    print("开始测试流式输出...\n")
    print(f"模型: {MODEL_NAME}")
    print(f"端口: {PORT}")
    print("-" * 50)

    try:
        # 4. 发起流式请求
        # response = client.chat.completions.create(
        #     model=MODEL_NAME,
        #     messages=[
        #         {"role": "user", "content": "请用一段话简单介绍一下量子计算。"}
        #     ],
        #     stream=True,  # 开启流式输出
        #     temperature=0.5,
        #     max_tokens=3000,
        # )

        # # 5. 实时打印返回的数据块 (chunks)
        # for chunk in response:
        #     content = chunk.choices[0].delta.content
        #     if content:
        #         # 使用 end="" 和 flush=True 确保文字能够逐字平滑显示
        #         print(content, end="", flush=True)

        response = client.completions.create(
            model=MODEL_NAME,  # 注意：必须使用支持补全接口的模型
            prompt="请用一段话简单介绍一下量子计算。", # 这里是纯字符串，不是 messages 列表
            stream=True,
            temperature=0.5,
            max_tokens=3000,
        )
        for chunk in response:
            if chunk.choices[0].text:
                print(chunk.choices[0].text, end="", flush=True)

    except Exception as e:
        print(f"\n[错误] 请求失败: {e}")

    print("\n" + "-" * 50)
    print("生成结束！")


if __name__ == "__main__":
    main()
