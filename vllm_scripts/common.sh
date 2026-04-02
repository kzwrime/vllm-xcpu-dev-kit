#!/bin/bash

# 函数：检查指定环境变量是否存在，存在则打印，不存在则报错退出。
# 参数: $1 - 要检查的环境变量名称
check_and_print_env() {
    local var_name="$1"
    # 使用 ${!var_name} 来间接引用变量的值
    if [ -z "${!var_name}" ]; then
        echo "❌ 错误：必需的环境变量 ${var_name} 未设置或为空。" >&2
        echo "    请设置 ${var_name}。" >&2
        exit 1
    else
        echo "✅ ${var_name} 已设置：'${!var_name}'"
    fi
}

# 函数：加载指定的环境变量文件。如果文件不存在则报错退出。
# 参数: $1 - 要加载的文件名 (相对于 $SCRIPT_DIR)
load_env_file() {
    local env_file="$1"

    if [ -f "$env_file" ]; then
        echo "✅ 正在加载环境文件: $env_file"
        # 使用 source 命令加载文件内容
        source "$env_file"

        # 检查 source 命令是否成功
        if [ $? -ne 0 ]; then
            echo "❌ 错误！加载文件 $env_file 失败。请检查文件内容是否有语法错误。" >&2
            exit 1
        fi
    else
        echo "❌ 错误！无法找到必需的环境文件: $env_file" >&2
        exit 1
    fi
}

# 函数：加载指定的预设文件
# 参数: $1 - 预设文件的完整路径
load_preset_file() {
    local preset_file="$1"

    if [ -f "$preset_file" ]; then
        echo "🚀 通过参数加载预设配置: $preset_file"
        source "$preset_file"
        if [ $? -ne 0 ]; then
            echo "❌ 错误！加载预设文件 $preset_file 失败。请检查文件内容是否有语法错误。" >&2
            exit 1
        fi
    else
        echo "❌ 错误！无法找到预设文件: $preset_file" >&2
        exit 1
    fi
}

# 函数：智能加载用户配置
# 优先级: user_env.sh > 预设配置 > user_env_template.sh
# 参数: 无
load_user_config() {
    local script_dir="$1"

    echo "load_user_config"

    # 优先级 1: 预设配置 (VLLM_CURRENT_PRESET 或 PRESET)
    local preset_value="${VLLM_CURRENT_PRESET:-${PRESET:-}}"
    if [ -n "$preset_value" ]; then
        local preset_file="$script_dir/presets/${preset_value}.sh"
        if [ -f "$preset_file" ]; then
            echo "🚀 优先加载预设配置: $preset_file"
            source "$preset_file"
        else
            echo "⚠️  警告：设置了预设但文件不存在: $preset_file"
            # 这里可以选择 exit 1 (强制报错) 或者继续尝试后续逻辑
        fi

    # 优先级 2: 用户自定义配置 (user_env.sh)
    elif [ -f "$script_dir/user_env.sh" ]; then
        echo "✅ 正在加载用户配置: $script_dir/user_env.sh"
        source "$script_dir/user_env.sh"

    # 优先级 3: 模板文件回退 (user_env_template.sh)
    elif [ -f "$script_dir/user_env_template.sh" ]; then
        echo "ℹ️  未检测到自定义配置，加载模板: $script_dir/user_env_template.sh"
        source "$script_dir/user_env_template.sh"

    # 最终兜底：报错退出
    else
        echo "❌ 错误！无法找到任何有效的配置文件。" >&2
        exit 1
    fi
}

# 函数：解析命令行参数并加载环境配置
# 用法: parse_args_and_load_env "vllm_scripts目录路径" "$@"
# 参数: $1 - vllm_scripts 目录的路径 (通常是 $SCRIPT_DIR/..)
#        $@ - 脚本的所有原始参数
parse_args_and_load_env() {
    local vllm_scripts_dir="$1"
    shift  # 移除第一个参数，保留原始的脚本参数

    # 解析命令行参数
    PRESET_FILE=""
    while getopts "e:" opt; do
        case $opt in
            e)
                PRESET_FILE="$OPTARG"
                ;;
            \?)
                echo "❌ 无效的选项: -$OPTARG" >&2
                echo "用法: $0 [-e 预设文件路径]" >&2
                echo "  -e  指定预设文件路径（相对于当前目录或绝对路径）" >&2
                exit 1
                ;;
        esac
    done

    # 检查是否有未处理的参数（可能是用户忘记加 -e）
    shift $((OPTIND-1))
    if [ $# -gt 0 ]; then
        echo "⚠️  警告：检测到额外的参数: $@" >&2
        echo "   如果要指定预设文件，请使用 -e 参数：" >&2
        echo "   $0 -e $1" >&2
        exit 1
    fi

    # 加载环境文件
    load_env_file "$vllm_scripts_dir/env.sh"

    # 如果通过 -e 参数指定了预设文件，直接加载；否则使用常规的 load_user_config 逻辑
    if [ -n "$PRESET_FILE" ]; then
        load_preset_file "$PRESET_FILE"
    else
        load_user_config "$vllm_scripts_dir"
    fi
}

