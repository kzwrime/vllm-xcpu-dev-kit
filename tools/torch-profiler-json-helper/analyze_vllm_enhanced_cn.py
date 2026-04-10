#!/usr/bin/env python3
"""
vLLM Trace 增强分析工具（中文版）

功能：
1. 支持按关键字过滤操作（如 allreduce, alltoallv）
2. 支持跳过前 N 次迭代或指定迭代范围
3. 自动处理 .json.gz 压缩文件
4. 四维分析：传统、方差、迭代、时序
5. 自动检测交替式负载不均
6. 中文输出，专有名词保留英文
"""

import json
import argparse
from collections import defaultdict
import statistics
import gzip
from pathlib import Path


def read_trace_file(trace_file):
    """
    Read trace file, automatically handling .json.gz compression.

    Args:
        trace_file: Path to trace file (.json or .json.gz)

    Returns:
        Parsed trace data (dict)
    """
    trace_file = Path(trace_file)

    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    # Handle .json.gz files
    if trace_file.suffix == ".gz":
        # Check if it's a .json.gz file
        if trace_file.stem.endswith('.json'):
            print(f"📦 读取压缩文件: {trace_file.name}")
            with gzip.open(trace_file, "rt") as f:
                data = json.load(f)
            print(f"   → 解压完成")
        else:
            # Plain .gz file
            with gzip.open(trace_file, "rt") as f:
                data = json.load(f)
    else:
        # Regular .json file
        with open(trace_file, "r") as f:
            data = json.load(f)

    return data


def analyze_vllm_trace_enhanced_cn(trace_file, top_n=20, show_iterations=True,
                                    keywords=None, case_sensitive=False,
                                    skip_iterations=0, skip_percent=0,
                                    from_iter=None, to_iter=None,
                                    count_iterations=None,
                                    iteration_ranks_per_line=4):
    """
    增强版分析（中文输出）

    Args:
        trace_file: Trace JSON 文件路径
        top_n: 显示前 N 个操作
        show_iterations: 是否显示迭代详情
        keywords: 关键字列表（只分析匹配的操作）
        case_sensitive: 关键字是否区分大小写
        skip_iterations: 跳过前 N 次迭代（warmup）
        skip_percent: 跳过前 N% 的迭代
        from_iter: 从第 N 次迭代开始分析（1-based）
        to_iter: 分析到第 N 次迭代（1-based）
        count_iterations: 只分析 N 次迭代
        iteration_ranks_per_line: [C] 逐迭代分析中每行最多显示多少个 Rank
    """

    # Read trace file (supports .json and .json.gz)
    try:
        data = read_trace_file(trace_file)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ 错误: 无法解析 JSON 文件: {e}")
        return
    except Exception as e:
        print(f"❌ 错误: 读取文件失败: {e}")
        return

    events = data.get("traceEvents", [])

    # 按进程和操作名分组事件
    processes = defaultdict(lambda: defaultdict(list))
    process_names = {}

    # 第一遍：收集进程名
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            pid = event.get("pid")
            name = event.get("args", {}).get("name", f"PID {pid}")
            process_names[pid] = name

    # 第二遍：收集持续时间事件
    for event in events:
        if event.get("ph") == "X":
            pid = event.get("pid")
            name = event.get("name", "unknown")
            duration = event.get("dur", 0)
            timestamp = event.get("ts", 0)

            # 关键字过滤
            if keywords:
                match = False
                search_name = name if case_sensitive else name.lower()
                for keyword in keywords:
                    search_keyword = keyword if case_sensitive else keyword.lower()
                    if search_keyword in search_name:
                        match = True
                        break
                if not match:
                    continue

            processes[pid][name].append({
                "duration": duration,
                "timestamp": timestamp,
                "event": event
            })

    # 按时间戳排序每个操作的事件
    for pid in processes:
        for op_name in processes[pid]:
            processes[pid][op_name].sort(key=lambda x: x["timestamp"])

    print("=" * 100)
    print(f"vLLM Trace 增强分析: {trace_file}")
    print("=" * 100)

    # 进程信息
    pids = sorted(processes.keys())
    print(f"\n检测到进程数: {len(pids)}")
    for pid in pids:
        proc_name = process_names.get(pid, f"PID {pid}")
        total_events = sum(len(v) for v in processes[pid].values())
        print(f"  {proc_name:20s} (PID {pid}): {total_events:10d} 个事件")

    # 过滤提示
    if keywords:
        print(f"\n关键字过滤: {', '.join(keywords)} (区分大小写: {'是' if case_sensitive else '否'})")

    # 获取所有操作并按总持续时间排序
    all_operations = {}
    for pid in pids:
        for op_name in processes[pid].keys():
            if op_name not in all_operations:
                all_operations[op_name] = 0
            all_operations[op_name] += sum(e["duration"] for e in processes[pid][op_name])

    sorted_ops = sorted(all_operations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    if not sorted_ops:
        print("\n未找到匹配的操作。请检查关键字或使用区分大小写模式。")
        return

    print("\n" + "=" * 100)
    print(f"Top {len(sorted_ops)} 操作 - 增强分析")
    print("=" * 100)

    # 计算迭代范围
    def calculate_iteration_range(total_iterations, skip_n, skip_pct, from_n, to_n, count_n):
        """
        计算实际要分析的迭代范围

        优先级：from_iter/to_iter > skip_iterations > skip_percent > count_iterations

        Returns:
            (start_idx, end_idx, description)
            start_idx 和 end_idx 是 0-based 的索引
        """
        start_idx = 0
        end_idx = total_iterations

        # from_iter 和 to_iter 有最高优先级
        if from_iter is not None or to_iter is not None:
            if from_iter is not None:
                start_idx = max(0, from_iter - 1)  # 转换为 0-based
            if to_iter is not None:
                end_idx = min(total_iterations, to_iter)

            desc = f"迭代范围: {from_iter if from_iter else 1} -> {to_iter if to_iter else total_iterations}"
            return (start_idx, end_idx, desc)

        # skip_iterations 次之
        if skip_n > 0:
            start_idx = min(skip_n, total_iterations)
            if count_n is not None:
                end_idx = min(total_iterations, start_idx + count_n)
                desc = f"跳过前 {skip_n} 次迭代，分析 {count_n} 次"
            else:
                desc = f"跳过前 {skip_n} 次迭代（warmup）"
            return (start_idx, end_idx, desc)

        # skip_percent 再次
        if skip_pct > 0:
            skip_n = int(total_iterations * skip_pct / 100)
            start_idx = min(skip_n, total_iterations)
            if count_n is not None:
                end_idx = min(total_iterations, start_idx + count_n)
                desc = f"跳过前 {skip_pct}% ({skip_n} 次) 迭代，分析 {count_n} 次"
            else:
                desc = f"跳过前 {skip_pct}% ({skip_n} 次) 迭代（warmup）"
            return (start_idx, end_idx, desc)

        # count_iterations 最后
        if count_n is not None:
            end_idx = min(count_n, total_iterations)
            desc = f"只分析前 {count_n} 次迭代"
            return (start_idx, end_idx, desc)

        # 默认：分析所有迭代
        desc = "分析所有迭代"
        return (start_idx, end_idx, desc)

    def chunk_list(items, chunk_size):
        if chunk_size <= 0:
            chunk_size = len(items) or 1
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    for rank, (op_name, total_duration) in enumerate(sorted_ops, 1):
        print(f"\n[{rank}] {op_name}")
        print("-" * 100)

        # 检查所有 Rank 的调用次数是否一致
        call_counts = [len(processes[pid].get(op_name, [])) for pid in pids]
        min_calls = min(call_counts)
        max_calls = max(call_counts)

        if min_calls == 0:
            print(f"  ⚠️  警告: 并非所有 Rank 都有此操作")
            continue

        if min_calls != max_calls:
            print(f"  ⚠️  警告: 调用次数不匹配 - 最小={min_calls}, 最大={max_calls}")
            num_iterations = min_calls
        else:
            num_iterations = min_calls

        # 计算迭代范围
        start_idx, end_idx, range_desc = calculate_iteration_range(
            num_iterations, skip_iterations, skip_percent,
            from_iter, to_iter, count_iterations
        )
        actual_iterations = end_idx - start_idx

        if start_idx > 0 or end_idx < num_iterations:
            print(f"  📊 迭代范围: {range_desc} (总共 {num_iterations} 次迭代，分析 {actual_iterations} 次)")
        else:
            print(f"  📊 迭代范围: 分析所有 {num_iterations} 次迭代")

        # === 分析 1: 传统分析（基于平均值）===
        print("\n  [A] 传统分析（基于平均时间）:")
        print("  " + "-" * 96)

        traditional_timings = []
        for pid in pids:
            if op_name in processes[pid]:
                # 应用迭代范围
                events_in_range = processes[pid][op_name][start_idx:end_idx]
                durations = [e["duration"] for e in events_in_range]
                total = sum(durations)
                avg = total / len(durations) if durations else 0
                count = len(durations)

                traditional_timings.append({"pid": pid, "total": total, "avg": avg, "count": count})

                proc_name = process_names.get(pid, f"PID {pid}")
                range_note = f" [范围 {start_idx+1}-{end_idx}]" if start_idx > 0 or end_idx < num_iterations else ""
                print(f"    {proc_name:20s}: 总计={total:15.0f} µs | 平均={avg:10.2f} µs | 次数={count:4d}{range_note}")

        # 传统不均衡度
        if len(traditional_timings) > 1:
            totals = [t["total"] for t in traditional_timings if t["total"] > 0]
            if totals:
                max_total = max(totals)
                min_total = min(totals)
                avg_total = sum(totals) / len(totals)

                if avg_total > 0:
                    imbalance_pct = (max_total - min_total) / avg_total * 100
                    ratio = max_total / min_total if min_total > 0 else float("inf")

                    if imbalance_pct > 20:
                        marker = "⚠️ 严重不均衡"
                    elif imbalance_pct > 10:
                        marker = "⚡ 中度不均衡"
                    else:
                        marker = "✓ 均衡"

                    print(f"    → 传统不均衡度: {imbalance_pct:6.2f}% (最大/最小 = {ratio:.2f}x) {marker}")

        # === 分析 2: 方差分析（Rank 内部稳定性）===
        print("\n  [B] 方差分析（Rank 内部稳定性）:")
        print("  " + "-" * 96)

        high_variance_ranks = []
        for pid in pids:
            if op_name in processes[pid]:
                # 应用迭代范围
                events_in_range = processes[pid][op_name][start_idx:end_idx]
                durations = [e["duration"] for e in events_in_range]

                if len(durations) > 1:
                    mean = statistics.mean(durations)
                    stdev = statistics.stdev(durations)
                    cv = (stdev / mean * 100) if mean > 0 else 0  # 变异系数
                else:
                    mean = durations[0] if durations else 0
                    stdev = 0
                    cv = 0

                proc_name = process_names.get(pid, f"PID {pid}")
                cv_marker = "⚠️ 高方差" if cv > 30 else ("⚡" if cv > 15 else "✓ 稳定")
                range_note = f" [{actual_iterations} 次迭代]" if start_idx > 0 or end_idx < num_iterations else ""
                print(f"    {proc_name:20s}: 均值={mean:10.2f} µs | 标准差={stdev:10.2f} µs | CV={cv:6.2f}% {cv_marker}{range_note}")

                if cv > 30:
                    high_variance_ranks.append(proc_name)

        if high_variance_ranks:
            print(f"    → ⚠️  警告: 检测到高方差: {', '.join(high_variance_ranks)}")

        # === 分析 3: 逐迭代不均衡度分析 ===
        if show_iterations and actual_iterations <= 50:
            print(f"\n  [C] 逐迭代不均衡度分析（显示 {actual_iterations} 次迭代，索引 {start_idx+1}-{end_idx}）:")
            print("  " + "-" * 96)

            per_iter_imbalances = []
            max_imbalance = 0
            max_imbalance_iter = -1

            for i in range(start_idx, end_idx):
                iter_durations = []
                iter_str_parts = []

                for pid in pids:
                    if op_name in processes[pid] and i < len(processes[pid][op_name]):
                        dur = processes[pid][op_name][i]["duration"]
                        iter_durations.append(dur)
                        proc_name = process_names.get(pid, f"PID {pid}")
                        iter_str_parts.append(f"{proc_name}={dur:6.0f}µs")

                if iter_durations:
                    max_dur = max(iter_durations)
                    min_dur = min(iter_durations)
                    avg_dur = sum(iter_durations) / len(iter_durations)

                    if avg_dur > 0:
                        imb = (max_dur - min_dur) / avg_dur * 100
                        per_iter_imbalances.append(imb)

                        if imb > max_imbalance:
                            max_imbalance = imb
                            max_imbalance_iter = i

                        marker = "⚠️" if imb > 50 else ("⚡" if imb > 20 else "✓")
                        iter_num = i + 1  # 转换为 1-based 显示
                        prefix = f"    迭代 {iter_num:7d}: "
                        chunks = chunk_list(iter_str_parts, iteration_ranks_per_line)

                        if not chunks:
                            continue

                        print(prefix + " | ".join(chunks[0]))
                        continuation_prefix = " " * len(prefix) + "  "
                        for chunk in chunks[1:-1]:
                            print(continuation_prefix + " | ".join(chunk))

                        if len(chunks) == 1:
                            print(f"{continuation_prefix}不均衡度={imb:6.2f}% {marker}")
                        else:
                            print(f"{continuation_prefix}{' | '.join(chunks[-1])} | 不均衡度={imb:6.2f}% {marker}")

            if per_iter_imbalances:
                avg_per_iter_imbalance = sum(per_iter_imbalances) / len(per_iter_imbalances)
                max_imbalance_val = max(per_iter_imbalances)

                print(f"\n    → 平均逐迭代不均衡度: {avg_per_iter_imbalance:6.2f}%")
                print(f"    → 最大逐迭代不均衡度: {max_imbalance_val:6.2f}% (迭代 {max_imbalance_iter + 1})")

                # 与传统方法对比
                traditional_imbalance = imbalance_pct if len(traditional_timings) > 1 else 0
                if avg_per_iter_imbalance > traditional_imbalance * 2:
                    print(f"    → ⚠️  严重: 逐迭代不均衡度 ({avg_per_iter_imbalance:.1f}%) >> 传统不均衡度 ({traditional_imbalance:.1f}%)")
                    print(f"       这表明存在交替式负载不均 - 不同迭代中不同的 Rank 表现较慢！")
                elif avg_per_iter_imbalance > traditional_imbalance * 1.5:
                    print(f"    → ⚡ 中度: 逐迭代不均衡度 ({avg_per_iter_imbalance:.1f}%) > 传统不均衡度 ({traditional_imbalance:.1f}%)")
                    print(f"       某些迭代的不均衡度高于平均值所暗示的水平。")
        else:
            if actual_iterations > 50:
                print(f"\n  [C] 逐迭代不均衡度分析: 跳过（分析迭代次数过多: {actual_iterations}）")
                print(f"      提示: 使用 --no-iterations 可隐藏此分析，或使用 --count 减少分析次数")

        # === 分析 4: 时间同步分析 ===
        print("\n  [D] 时间同步分析（时间偏差分析）:")
        print("  " + "-" * 96)

        has_skew = False
        # 检查范围内的前几次迭代
        check_iterations = min(5, actual_iterations)
        for iter_offset in range(check_iterations):
            i = start_idx + iter_offset  # 实际的全局索引
            timestamps = []
            for pid in pids:
                if op_name in processes[pid] and i < len(processes[pid][op_name]):
                    timestamps.append({
                        "pid": pid,
                        "ts": processes[pid][op_name][i]["timestamp"],
                        "dur": processes[pid][op_name][i]["duration"]
                    })

            if timestamps:
                min_ts = min(t["ts"] for t in timestamps)
                max_ts = max(t["ts"] for t in timestamps)
                time_skew = max_ts - min_ts

                iter_num = i + 1
                if time_skew > 100:  # 超过 0.1ms 偏差
                    print(f"    迭代 {iter_num}: 时间偏差 = {time_skew:.0f} µs ⚠️")
                    has_skew = True
                else:
                    if time_skew > 0:
                        print(f"    迭代 {iter_num}: 时间偏差 = {time_skew:.0f} µs ✓")
                    break  # 只显示第一次有偏差的迭代

        if not has_skew and actual_iterations > 0:
            print(f"    前 {check_iterations} 次迭代时间同步良好 ✓")

        print("\n" + "=" * 100)

    # 总体总结
    print("\n" + "=" * 100)
    print("总体总结")
    print("=" * 100)
    print("\n关键发现:")
    print("  1. 传统分析显示所有 Rank 的平均负载分布")
    print("  2. 方差分析检测性能不稳定的 Rank（高 CV）")
    print("  3. 逐迭代分析揭示交替式负载不均模式")
    print("  4. 时间分析识别 Rank 之间的时间偏差")
    print("\n优化建议:")
    print("  - 高方差 per Rank: 检查 CPU 绑定、频率缩放或后台进程")
    print("  - 高逐迭代不均衡度: 调查动态负载分配或缓存效应")
    print("  - 大时间偏差: 检查同步原语和网络延迟")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Trace 增强分析工具（中文版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础分析（所有操作）
  python analyze_vllm_enhanced_cn.py trace.json

  # 跳过前 10 次迭代（warmup）
  python analyze_vllm_enhanced_cn.py trace.json --skip 10

  # 跳过前 20% 的迭代
  python analyze_vllm_enhanced_cn.py trace.json --skip-percent 20

  # 只分析前 5 次迭代
  python analyze_vllm_enhanced_cn.py trace.json --count 5

  # 从第 50 次迭代开始，分析 5 次
  python analyze_vllm_enhanced_cn.py trace.json --from-iter 50 --count 5

  # 分析第 50-100 次迭代
  python analyze_vllm_enhanced_cn.py trace.json --from-iter 50 --to-iter 100

  # Rank 很多时，逐迭代详情每行显示 8 个 Rank
  python analyze_vllm_enhanced_cn.py trace.json --count 3 --iteration-ranks-per-line 8

  # 只分析通信操作（关键字过滤）
  python analyze_vllm_enhanced_cn.py trace.json -k allreduce alltoallv

  # 组合使用：跳过前 10% 并分析 20 次迭代
  python analyze_vllm_enhanced_cn.py trace.json --skip-percent 10 --count 20 -k allreduce
        """
    )

    parser.add_argument("trace_file", help="要分析的 Trace JSON 文件")
    parser.add_argument("--top", "-t", type=int, default=20,
                        help="显示前 N 个操作（默认: 20）")
    parser.add_argument("--no-iterations", action="store_true",
                        help="隐藏逐迭代详情（适用于大型 trace）")

    # 关键字过滤
    parser.add_argument("--keywords", "-k", nargs="+", metavar="KEYWORD",
                        help="只分析包含这些关键字的操作（例如: allreduce alltoallv）")
    parser.add_argument("--case-sensitive", "-c", action="store_true",
                        help="关键字匹配区分大小写（默认: 不区分）")

    # 迭代范围控制
    iter_group = parser.add_argument_group("迭代范围控制", "控制分析哪些次迭代（用于跳过 warmup 或聚焦特定阶段）")
    iter_group.add_argument("--skip", type=int, metavar="N",
                           help="跳过前 N 次迭代（warmup 阶段）")
    iter_group.add_argument("--skip-percent", type=int, metavar="P",
                           help="跳过前 P%% 的迭代（例如: 20 表示跳过前 20%%）")
    iter_group.add_argument("--count", type=int, metavar="N",
                           help="只分析 N 次迭代（从起始位置开始）")
    iter_group.add_argument("--from-iter", type=int, metavar="N",
                           help="从第 N 次迭代开始分析（1-based）")
    iter_group.add_argument("--to-iter", type=int, metavar="N",
                           help="分析到第 N 次迭代（1-based，包含）")
    iter_group.add_argument("--iteration-ranks-per-line", type=int, metavar="N", default=4,
                           help="逐迭代不均衡度分析中每行最多显示多少个 Rank（默认: 4）")

    args = parser.parse_args()

    # 验证参数
    if args.skip is not None and args.skip < 0:
        parser.error("--skip 必须是非负整数")
    if args.skip_percent is not None and (args.skip_percent < 0 or args.skip_percent > 100):
        parser.error("--skip-percent 必须在 0-100 之间")
    if args.count is not None and args.count < 1:
        parser.error("--count 必须是正整数")
    if args.from_iter is not None and args.from_iter < 1:
        parser.error("--from-iter 必须是正整数（1-based）")
    if args.to_iter is not None and args.to_iter < 1:
        parser.error("--to-iter 必须是正整数（1-based）")
    if args.from_iter is not None and args.to_iter is not None and args.from_iter > args.to_iter:
        parser.error("--from-iter 不能大于 --to-iter")
    if args.iteration_ranks_per_line < 1:
        parser.error("--iteration-ranks-per-line 必须是正整数")

    # 检查参数冲突
    if (args.from_iter is not None or args.to_iter is not None) and (args.skip or args.skip_percent):
        parser.error("--from-iter/--to-iter 不能与 --skip/--skip-percent 同时使用")

    analyze_vllm_trace_enhanced_cn(
        args.trace_file,
        args.top,
        not args.no_iterations,
        args.keywords,
        args.case_sensitive,
        args.skip or 0,
        args.skip_percent or 0,
        args.from_iter,
        args.to_iter,
        args.count,
        args.iteration_ranks_per_line
    )


if __name__ == "__main__":
    main()
