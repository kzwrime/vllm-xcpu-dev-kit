#!/usr/bin/env python3
"""
准确对比 vLLM 实际运行和 benchmark 的参数
"""

print("=" * 80)
print(" vLLM 实际运行 vs Benchmark 参数对比")
print("=" * 80)

# vLLM 实际运行数据（从日志提取）
print("\n## vLLM 实际运行数据（28 tokens）\n")
print("rotary_embedding 调用:")
print("  positions: [28]")
print("  query: [28, 2048]     # [28, 16*128]")
print("  key: [28, 1024]       # [28, 8*128]")
print("  head_size: 128        # ← 实际是 128，不是 64！")
print("  cos_sin_cache: [40960, 128]")
print("  is_neox: True")
print("  dtype: bfloat16")

# Benchmark 数据（修复后）
print("\n## Benchmark 数据（7 tokens，修复后）\n")
print("rotary_embedding 调用:")
print("  positions: [7]")
print("  query: [7, 2048]      # [7, 16*128]")
print("  key: [7, 1024]        # [7, 8*128]")
print("  head_size: 128")
print("  cos_sin_cache: [8192, 64]")
print("  is_neox: True")
print("  dtype: float32")

# 对比分析
print("\n## 对比分析\n")

matches = []
differences = []

# Query 形状
print("1. Query 形状:")
print("   vLLM: [28, 2048]")
print("   Benchmark: [7, 2048]")
print("   → 维度比例正确（第二个维度都是 2048 = 16*128）✓")
matches.append("Query 形状")

# Key 形状
print("\n2. Key 形状:")
print("   vLLM: [28, 1024]")
print("   Benchmark: [7, 1024]")
print("   → 维度比例正确（第二个维度都是 1024 = 8*128）✓")
matches.append("Key 形状")

# head_size 参数
print("\n3. head_size 参数:")
print("   vLLM: 128")
print("   Benchmark: 128")
print("   → 完全匹配 ✓")
matches.append("head_size 参数")

# cos_sin_cache 形状
print("\n4. cos_sin_cache 形状:")
print("   vLLM: [40960, 128]")
print("   Benchmark: [8192, 64]")
print("   → 不匹配 ❌")
print("   分析:")
print("     - 第一个维度：40960 vs 8192（可能是 max_position 不同）")
print("     - 第二个维度：128 vs 64（rotary_dim 不同）")
differences.append("cos_sin_cache 形状")

# 数据类型
print("\n5. 数据类型:")
print("   vLLM: bfloat16")
print("   Benchmark: float32")
print("   → 不同（但这是测试配置问题，不是形状问题）")
differences.append("dtype")

# 总结
print("\n" + "=" * 80)
print(" 总结")
print("=" * 80)

print(f"\n✓ 匹配项 ({len(matches)}):")
for i, item in enumerate(matches, 1):
    print(f"  {i}. {item}")

print(f"\n❌ 差异项 ({len(differences)}):")
for i, item in enumerate(differences, 1):
    print(f"  {i}. {item}")

print("\n关键发现:")
print("  1. ✓ Query/Key 形状现在是正确的（GQA 支持）")
print("  2. ✓ head_size 参数正确（128）")
print("  3. ❌ cos_sin_cache 的 rotary_dim 不同:")
print("      - vLLM 使用 128（全旋转）")
print("      - Benchmark 配置使用 64（部分旋转）")
print("  4. ℹ️  需要验证 Qwen3-0.6B 的实际 rotary_dim 配置")

print("\n建议:")
print("  1. 检查 Qwen3-0.6B 模型配置，确认实际的 rotary_dim")
print("  2. 如果实际是 128，需要更新 model_configs.py")
print("  3. cos_sin_cache 的第一个维度（max_position）差异可以忽略")

print("\n" + "=" * 80)
