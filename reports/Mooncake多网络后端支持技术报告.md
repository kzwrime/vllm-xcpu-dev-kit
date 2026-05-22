# Mooncake 多网络后端支持技术报告

**日期**：2026-05-22  
**范围**：基于 `deps/Mooncake/README.md` 与 `deps/Mooncake/mooncake-transfer-engine` 源码，整理 Mooncake 框架、能力、性能、后端支持，并分析 TCP/RDMA 网络后端设计及新增定制网卡后端的接入路径。

## README 内容译介：框架、能力、性能与后端

本节翻译并整理 `deps/Mooncake/README.md` 中与框架介绍、系统能力、性能数据和后端支持相关的主要内容，作为后续源码分析的背景。

### 项目定位

Mooncake 是面向 LLM Serving 的、以 KVCache 为中心的解耦式架构。该项目是 Kimi 服务背后的 serving 平台之一，仓库中开源了 Transfer Engine、Mooncake Store，并包含技术报告与开源 trace。

Mooncake 的核心目标是把 prefill 集群和 decode 集群解耦，并利用 GPU 集群中相对未充分使用的 CPU、DRAM、SSD 等资源构建分布式 KVCache 池。其调度器以 KVCache 为中心，在满足延迟 SLO 的前提下最大化整体有效吞吐。README 中给出的实验结论包括：在长上下文场景中，Mooncake 相比 baseline 在部分模拟场景下可实现最高 525% 的吞吐提升；在真实工作负载下，该架构使 Kimi 能处理约 75% 更多请求。

### 核心组件

| 组件 | README 描述整理 | 与本文网络后端分析的关系 |
|---|---|---|
| Transfer Engine (TE) | Mooncake 的核心组件，提供跨存储设备和网络链路的批量数据传输统一接口；支持 TCP、RDMA、CXL/shared-memory、NVMe over Fabric 等协议 | 本文重点分析 TE 中的 `Transport` 抽象、TCP/RDMA 后端以及新增后端接入方式 |
| P2P Store | 基于 Transfer Engine 构建，用于集群节点之间共享临时对象，例如 checkpoint 文件，避免单机带宽成为瓶颈 | 展示 TE 作为通用数据搬运层的上层用法 |
| Mooncake Store | 基于 Transfer Engine 的分布式 KVCache 存储引擎，面向 XpYd / prefill-decode 解耦，提供可复用 KV cache 的跨位置存储 | 依赖 TE 的多后端能力实现远程 KVCache 访问 |
| LLM 推理系统集成 | 已与 vLLM、SGLang、LMCache 等系统集成，支持 prefill-decode disaggregation、HiCache、prefill serving 等场景 | 多网络后端直接影响跨节点 KVCache/embedding/hidden states 传输效率 |
| Elastic Expert Parallelism | 为 MoE 推理增加弹性和容错能力，支持故障 rank 检测，并可与 EPLB 模块配合把 token 路由到健康 rank | 属于 Mooncake 更上层的弹性计算能力 |
| Tensor-Centric Ecosystem | 以 Tensor 作为基础数据载体，覆盖 Transfer Engine、P2P Store、Mooncake Store 到 Mooncake Backend | 说明 TE 不只服务 KVCache，也可服务 checkpoint、hidden states、Tensor 对象等数据 |

### Transfer Engine 能力

README 将 Transfer Engine 描述为高性能数据传输框架：它提供统一接口在 DRAM、VRAM、NVMe 等存储介质之间搬运数据，同时隐藏与硬件相关的细节。其支持的通信/传输协议包括：

- TCP
- RDMA，包括 InfiniBand、RoCEv2、eRDMA、NVIDIA GPUDirect RDMA
- AWS EFA
- NVMe over Fabric (NVMe-oF)
- NVLink / intra-node NVLink / multi-node NVLink
- HIP
- Barex
- CXL
- Ascend 系列传输路径

README 强调的 TE 特性包括：

| 能力 | 说明 |
|---|---|
| 多 RDMA 网卡聚合 | 支持同时使用多张 RDMA NIC，实现传输带宽聚合 |
| 拓扑感知路径选择 | 根据源端和目标端内存位置、NUMA 亲和性等信息选择更合适的设备 |
| 临时网络错误恢复 | 传输失败时可自动尝试替代路径，提升临时故障下的鲁棒性 |
| 异构内存识别和路由 | 在启用对应 runtime 时，可识别 CUDA、MUSA、HIP、MACA、Cambricon MLU、Ascend 等环境中的加速器内存 |

### 性能数据

README 给出 Transfer Engine 的性能样例：以 40GB 数据为例，该数据量相当于 LLaMA3-70B 模型在 128k tokens 下产生的 KVCache 大小。在 4×200Gbps 和 8×400Gbps RoCE 网络中，Mooncake Transfer Engine 分别达到最高约 87GB/s 和 190GB/s 带宽，相比 TCP 协议约快 2.4 倍和 4.6 倍。

README 还给出 vLLM 集成中的 TTFT 对比：在支持 topology-aware path selection 与多网卡带宽聚合后，使用 Transfer Engine (RDMA) 的 vLLM 平均 TTFT 低于传统 TCP 传输。表中数据为：

| Backend / Setting | Output Token Throughput (tok/s) | Total Token Throughput (tok/s) | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| Transfer Engine (RDMA) | 12.06 | 2042.74 | 1056.76 | 635.00 | 4006.59 |
| TCP | 12.05 | 2041.13 | 1414.05 | 766.23 | 6035.36 |

该结果说明，在吞吐接近的情况下，RDMA 后端通过更低的数据传输延迟改善了 TTFT。README 也指出后续可通过 GPUDirect RDMA 和 zero-copy 继续改进 TTFT。

### 支持硬件与传输后端

README 将 Mooncake 的硬件和 transport 支持分为加速器 runtime、网络/fabric、专用传输路径三类。

加速器 runtime：

| 平台 | README 中的支持方式 |
|---|---|
| Huawei Ascend | `-DUSE_ASCEND=ON`、`-DUSE_ASCEND_DIRECT=ON`、`-DUSE_UBSHMEM=ON`、`-DUSE_ASCEND_HETEROGENEOUS=ON`；覆盖 HCCL、Ascend Direct、UBShmem、异构 Ascend-GPU transport |
| Cambricon MLU | `-DUSE_MLU=ON`；MLU 内存检测、拓扑发现和注册复用标准 `rdma` 数据路径 |
| Moore Threads MUSA | `-DUSE_MUSA=ON` |
| MetaX MACA | `-DUSE_MACA=ON` |
| T-Head PPU / Barex | 通过 Barex-based transport 支持 |
| NVIDIA CUDA / NVLink | `-DUSE_CUDA=ON`、`-DUSE_INTRA_NVLINK=ON`、`-DUSE_MNNVL=ON`；覆盖 CUDA memory、GPUDirect RDMA、GPUDirect Storage、intra-node NVLink、multi-node NVLink |
| AMD ROCm / HIP | `-DUSE_HIP=ON` |
| Hygon DCU / DTK | `-DUSE_HYGON=ON` |
| Iluvatar CoreX | `-DUSE_COREX=ON` |

网络与 fabric：

| Fabric / Transport | README 中的支持方式 |
|---|---|
| Alibaba Cloud eRDMA | 走标准 `rdma` 数据路径，可使用 `erdma_0` 等设备，并可启用 `CONFIG_ERDMA` |
| InfiniBand / RoCE | 走标准 `rdma` 协议路径，支持 topology-aware NIC selection |
| AWS EFA | `-DUSE_EFA=ON`，基于 libfabric SRD 的 EFA transport |
| NVMe-oF | `-DUSE_NVMEOF=ON` |
| CXL | `-DUSE_CXL=ON` |
| TCP/IP | `tcp`，作为通用 baseline，在普通网络环境中可用 |

专用传输路径：

| Transport path | README 中的支持方式 |
|---|---|
| Ascend HCCL transport | `-DUSE_ASCEND=ON` |
| Ascend Direct transport | `-DUSE_ASCEND_DIRECT=ON` |
| UBShmem transport | `-DUSE_UBSHMEM=ON`，示例支持 `--protocol=ubshmem` |
| Heterogeneous Ascend transport | `-DUSE_ASCEND_HETEROGENEOUS=ON` |
| Barex transport | `-DUSE_BAREX=ON`，高级 transport |
| Sunrise Transport | README 将其列为专用传输路径之一 |

### 上层使用场景

| 场景 | README 内容整理 |
|---|---|
| Transfer Engine standalone | 用于高性能 DRAM/VRAM/NVMe 数据传输，隐藏底层硬件细节 |
| P2P Store | 用于集群节点间临时对象共享，适合 checkpoint 传输，采用客户端架构并用 etcd 管理全局 metadata |
| Mooncake Store | 分布式 KVCache 存储引擎，支持多副本、大对象 striping 和并行 I/O，充分利用多 NIC 聚合带宽 |
| SGLang Integration | 作为 HiCache storage backend，支持分层 KV cache、write-through/write-back 策略、prefetch、page-first layout、zero-copy、GPU-assisted I/O、layer-wise overlap 等优化 |
| vLLM Integration | 用 Transfer Engine 替代默认 `nccl/gloo` 网络层，支持跨节点 KVCache 传输和 prefill-decode disaggregation |

### 使用与构建要点

README 对使用路径的描述包括：

- Python wheel：CUDA 系统可安装 `mooncake-transfer-engine` 或 CUDA 13 对应包；非 CUDA 系统可安装 `mooncake-transfer-engine-non-cuda`。
- Docker：支持从源码构建 wheel 并安装，示例中强调 GPU/RDMA、host network、IPC、memlock、hugepage 等部署参数。
- 源码构建：需要 gcc/g++、cmake、Go；可选 CUDA、Neuware、DTK、CoreX、Rust、hiredis、curl 等依赖；标准流程是 `mkdir build && cmake .. && make -j`。

README 也指出 Mooncake 为高速 RDMA 网络设计和优化。虽然支持 TCP-only 数据传输，但进行功能和性能评估时建议使用 RDMA 网络。

## 目录

1. [背景与动机](#1-背景与动机)
2. [整体架构](#2-整体架构)
3. [核心抽象与路由机制](#3-核心抽象与路由机制)
4. [TCP 后端源码分析](#4-tcp-后端源码分析)
5. [RDMA 后端源码分析](#5-rdma-后端源码分析)
6. [新增定制网卡后端的最小路径](#6-新增定制网卡后端的最小路径)
7. [最简测试方案](#7-最简测试方案)
8. [理解验证与质量检查](#8-理解验证与质量检查)
9. [实验验证结果](#9-实验验证结果)
10. [一页总结](#10-一页总结)

## 1. 背景与动机

Mooncake 的 Transfer Engine 把远端内存访问抽象成“Segment + BatchTransfer”：每个进程发布一个本地 segment，segment 内挂载已注册 buffer；发起端打开目标 segment 后提交 READ/WRITE 批量请求。网络后端只负责把请求真正落到 TCP、RDMA、EFA、NVLink、CXL 等具体介质上。

**WHY 需要多后端**：推理 KV cache、分布式对象存储、GPU/CPU 混合内存传输面对的硬件差异很大。只有 TCP 会牺牲低延迟和零拷贝；只有 RDMA 又无法覆盖标准 IP 网络、云厂商特殊网络、定制网卡。多后端让同一层 API 在“可用性”和“高性能”之间切换。

**WHY 这样设计**：Mooncake 没有把后端能力散落在业务代码里，而是把共同生命周期收敛到 `Transport` 基类，把协议选择放在 `MultiTransport`，把 segment/buffer/握手信息放在 `TransferMetadata`。新增后端时，应用层不需要理解网卡协议，只要目标 segment 的 `protocol` 能路由到已安装 transport。

【图示：多后端分层架构】

## 2. 整体架构

```
应用 / Mooncake Store / Python Binding
        |
TransferEngine / TransferEngineImpl
        |
MultiTransport
        |
Transport 抽象: install / registerLocalMemory / submitTransferTask / getTransferStatus
        |
tcp / rdma / efa / barex / cxl / nvlink / hip / ascend ...
        |
TransferMetadata: segment desc, buffer desc, rpc meta, handshake, notify
```

源码地图：

| 模块 | 关键文件 | 职责 |
|---|---|---|
| 公共抽象 | [transport.h](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/include/transport/transport.h:345) | `Transport` 虚接口、`BatchDesc`、`TransferTask`、`Slice` |
| 多后端管理 | [multi_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:305) | 后端构造、安装、按目标 segment 协议路由 |
| 引擎生命周期 | [transfer_engine_impl.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp:120) | 初始化 metadata/RPC，自动或手动安装 transport，广播内存注册 |
| 元数据 | [transfer_metadata.h](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/include/transfer_metadata.h:34) | segment、buffer、device、handshake、notify 描述 |
| TCP | [tcp_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:520) | TCP 数据端口、socket 会话、异步读写 |
| RDMA | [rdma_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:93) | verbs 资源、MR 注册、切片、设备选择 |
| RDMA worker | [worker_pool.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp:60) | 远端 rkey 查询、endpoint 建连、post send、poll CQ、retry |
| 端到端验证 | [transfer_engine_validator.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/example/transfer_engine_validator.cpp:394) | target/initiator 双进程数据一致性验证 |

## 3. 核心抽象与路由机制

### 3.1 Transport 是后端插件接口

`Transport` 的关键虚函数包括：

| 接口 | 必须性 | 新后端要做什么 |
|---|---:|---|
| `install(local_server_name, metadata, topo)` | 必须 | 初始化网卡资源，创建本地 segment，启动控制面或数据面服务，发布 metadata |
| `registerLocalMemory(addr, length, location, remote_accessible, update_metadata)` | 必须 | 注册本地 buffer，填充后端需要的 key/handle/address，并写入 `BufferDesc` |
| `unregisterLocalMemory` | 必须 | 释放本地注册资源并删除 metadata buffer |
| `registerLocalMemoryBatch` | 必须 | 批量注册，通常延迟一次 `updateLocalSegmentDesc()` |
| `submitTransferTask(vector<TransferTask*>)` | 实际必须 | `MultiTransport` 会按后端聚合任务后调用它 |
| `getTransferStatus(batch_id, task_id, status)` | 必须 | 汇总 slice 完成/失败计数，返回 `WAITING/COMPLETED/FAILED` |

`Transport` 同时定义了 `TransferTask` 和 `Slice`。`TransferTask` 保存原始请求、总字节数、slice 列表和完成计数；`Slice` 是后端实际提交的最小 I/O 单元，union 内已有 `rdma/tcp/local/nvmeof/cxl/...` 字段。见 [transport.h](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/include/transport/transport.h:306)。

### 3.2 MultiTransport 是后端工厂和路由器

安装路径在 [multi_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:305)：`proto == "rdma"` 创建 `RdmaTransport`，`proto == "tcp"` 创建 `TcpTransport`，其他协议在编译宏打开后加入。安装成功后保存到 `transport_map_[proto]`。

提交路径是：

1. `TransferEngineImpl::submitTransfer()` 调用 `multi_transports_->submitTransfer()`。
2. `MultiTransport` 对每个请求读取目标 `SegmentDesc`。
3. 用 `target_segment_desc->protocol` 选择 transport。
4. 把属于同一 transport 的 `TransferTask*` 聚合后调用 `submitTransferTask()`。

路由逻辑见 [multi_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:432)。多协议编译模式下，segment 的 `protocol` 可以是逗号分隔列表，`mp_selectTransport()` 会检查调用者指定的 preferred proto 是否被目标 segment 支持。

【图示：请求路由流程】

### 3.3 Metadata 是后端间的控制面契约

`SegmentDesc` 至少要包含：

| 字段 | TCP | RDMA | 定制网卡后端参考 |
|---|---|---|---|
| `name` | segment 名 | segment 名 | 必须 |
| `protocol` | `"tcp"` | `"rdma"` | 新协议名，如 `"customnic"` |
| `buffers` | addr/length | addr/length/lkey/rkey/location | 自定义 remote key/handle |
| `devices/topology` | 不需要 | NIC 列表与拓扑选择 | 如果多网卡或 NUMA 相关，建议需要 |
| `tcp_data_port` | TCP 数据端口 | 不需要 | 可新增端口或复用 `RpcMetaDesc`/handshake |

新增后端的核心不是“写一个发送函数”这么简单，而是定义清楚本地 buffer 注册后如何被远端寻址、远端需要什么 key、建连握手如何交换网卡上下文。

## 4. TCP 后端源码分析

TCP 是最简单的可用后端，适合作为定制网卡后端的第一份模板。

### 4.1 安装和元数据发布

`TcpTransport::install()` 做四件事：

1. 找一个可用 TCP 数据端口。
2. `allocateLocalSegmentID(tcp_port)` 创建/更新本地 segment。
3. 启动 metadata handshake daemon。
4. 发布本地 segment，并启动 asio accept 线程。

对应源码见 [tcp_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:520)。`allocateLocalSegmentID()` 把 `protocol` 设置为 `tcp`，并写入 `tcp_data_port`，见 [tcp_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:560)。

### 4.2 内存注册

TCP 没有真实 MR，也没有 rkey/lkey。`registerLocalMemory()` 只把 `addr/length` 写进 `BufferDesc`，多协议模式下额外标注 `buffer_desc.protocol = "tcp"`，见 [tcp_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:576)。

这个设计说明：Mooncake 的“内存注册”不等同于硬件 pin memory。对普通后端，它可以只是发布可访问地址范围；对 RDMA/定制网卡，它才需要驱动级注册。

### 4.3 数据面

TCP 请求不分多片，每个 request 生成一个 slice：

```cpp
slice->source_addr = request.source;
slice->length = request.length;
slice->opcode = request.opcode;
slice->tcp.dest_addr = request.target_offset;
slice->target_id = request.target_id;
startTransfer(slice);
```

见 [tcp_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:670)。

`startTransfer()` 读取目标 segment 与 RPC meta，连接目标 `tcp_data_port`，创建 `ClientSession`。会话头 `SessionHeader` 携带 `size/addr/opcode`；READ 表示 server 从目标地址写回数据，WRITE 表示 server 读取客户端发来的 body 写入目标地址。完成后调用 `slice->markSuccess()`，失败调用 `slice->markFailed()`，见 [tcp_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:875)。

### 4.4 TCP 底层调用接口

TCP 后端底层不是直接调用 Linux `send(2)/recv(2)`，而是通过 standalone Asio 的 TCP socket API 封装异步 I/O。头文件 [tcp_transport.h](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/include/transport/tcp_transport/tcp_transport.h:33) 引入 `<asio/ip/tcp.hpp>`，核心类型是 `asio::ip::tcp::socket`、`asio::ip::tcp::acceptor`、`asio::io_context`。

服务端监听路径：

1. `TcpContext(short port)` 构造 `asio::ip::tcp::endpoint`。
2. 优先打开 IPv6 dual-stack listener：`acceptor.open()`、`set_option(asio::ip::v6_only(false))`、`set_option(reuse_address(true))`、`bind()`、`listen()`。
3. 如果 dual-stack 失败，回退到 IPv4 endpoint。
4. `doAccept()` 调用 `acceptor.async_accept(...)`，为每个连接创建 `ServerSession`。
5. `TcpTransport::worker()` 循环调用 `context_->doAccept()` 和 `context_->io_context.run()`。

客户端连接路径：

1. `getConnection(host, port)` 使用 `asio::ip::tcp::resolver` 解析目标地址。
2. 创建 `std::shared_ptr<asio::ip::tcp::socket>`。
3. 调用同步 `asio::connect(*socket, endpoint_iterator)` 建立连接。
4. 如果启用 `MC_TCP_ENABLE_CONNECTION_POOL`，连接会进入 `connection_pool_`，完成后 `returnConnection()` 标记为空闲；否则每次传输完成后关闭 socket。

数据收发路径：

| 阶段 | 调用接口 | 说明 |
|---|---|---|
| 客户端发 header | `asio::async_write(socket, asio::buffer(&header_, sizeof(SessionHeader)), cb)` | header 包含目标地址、长度、READ/WRITE |
| 客户端 WRITE 发送 body | `asio::async_write(socket, asio::buffer(dram_buffer, buffer_size), cb)` | 分块发送本地 buffer |
| 客户端 READ 接收 body | `asio::async_read(socket, asio::buffer(dram_buffer, buffer_size), cb)` | 分块读回远端数据 |
| 服务端读 header | `asio::async_read(socket, asio::buffer(&header_, sizeof(SessionHeader)), cb)` | 根据 opcode 选择读 body 或写 body |
| 服务端 READ 回包 | `asio::async_write(socket, asio::buffer(dram_buffer, buffer_size), cb)` | 从目标地址读出并发回 |
| 服务端 WRITE 落地 | `asio::async_read(socket, asio::buffer(dram_buffer, buffer_size), cb)` | 读入客户端 body 后写到目标地址 |

如果编译时启用了 CUDA/MUSA/HIP/MLU/MACA 等宏，TCP 后端还会用 `cudaPointerGetAttributes()` 判断地址是否为设备内存；设备内存场景会先用 `cudaMemcpy(..., cudaMemcpyDefault)` 在设备内存和临时 DRAM buffer 之间搬运，再交给 Asio socket。因此 TCP 后端在 GPU buffer 上不是零拷贝路径，它是“设备内存 staging + TCP socket”的兼容实现。

### 4.5 TCP 的设计启示

| 设计点 | 对新增后端的启示 |
|---|---|
| 控制面简单，数据面自带连接 | 如果定制网卡也有 socket-like API，可以先照 TCP 做 |
| 不需要 topology | 单网卡或无需亲和性时可以先不接 topology |
| 每请求一个 slice | 最小实现可不切片，先跑通功能 |
| 完成语义统一 | 无论底层异步机制如何，最终必须调用 `markSuccess/markFailed` |

【图示：TCP 后端收发流程】

## 5. RDMA 后端源码分析

RDMA 是高性能后端模板，展示了 Mooncake 支持多网卡、拓扑选择、远端 key、连接池、失败重试的完整形态。

### 5.1 安装阶段

`RdmaTransport::install()` 要求传入 topology，初始化 RDMA resources，创建本地 segment，启动 handshake daemon，再发布 segment，见 [rdma_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:93)。

`initializeRdmaResources()` 遍历 topology 中的 HCA，为每张可用 RNIC 构造 `RdmaContext`。失败设备会被禁用；如果没有可用 RNIC，则安装失败，见 [rdma_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:659)。

### 5.2 内存注册

RDMA 注册比 TCP 复杂：

1. 可选 pre-touch 大内存，降低后续注册成本。
2. 对每个 `RdmaContext` 调用 `registerMemoryRegion()`。
3. 收集每个 context 对应的 `lkey/rkey`。
4. 根据 `location` 或地址自动识别 memory location。
5. 把 `addr/length/name/lkey/rkey/protocol` 写入 metadata。

这段路径从 [rdma_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:175) 开始。`RdmaContext` 内部调用 ibverbs 注册 MR，并维护地址到 MR 的 map，见 [rdma_context.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp:320)。

### 5.3 数据切片与本地设备选择

`submitTransferTask()` 会按 `globalConfig().slice_size` 切分请求。每个 slice 选择本地 buffer 与 device，写入本地 `source_lkey`，按 `RdmaContext` 聚合，然后提交给 worker pool，见 [rdma_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:464)。

关键点：本地设备选择依赖本地 segment 的 buffer location 与 topology。没有找到已注册且 active 的设备时，返回 `AddressNotRegistered`。

### 5.4 远端 rkey、endpoint 和完成队列

`WorkerPool::submitPostSend()` 再读取目标 segment，按目标地址选择远端 buffer/device，取出远端 `rkey`，构造 `peer_nic_path`，然后按 shard 投递到 worker 队列，见 [worker_pool.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp:60)。

`performPostSend()` 获取或创建 endpoint；如果 endpoint 未连接，则主动发起 handshake；连接后调用 `endpoint->submitPostSend()`，见 [worker_pool.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp:173)。`RdmaEndPoint` 最终把 slice 转成 `ibv_send_wr`，设置 RDMA READ/WRITE、local lkey、remote addr、remote rkey 并调用 `ibv_post_send()`。

完成由 CQ polling 驱动：成功 WC 调用 `slice->markSuccess()`；失败时删除 endpoint、增加 retry 并尝试 redispatch 到其他路径，超过重试次数后 `markFailed()`。

### 5.5 RDMA 的设计启示

| 设计点 | 对定制网卡后端的启示 |
|---|---|
| `Context` 表示本地网卡资源 | 定制网卡也建议抽出 `CustomContext` 管 device/queue/key |
| `Endpoint` 表示本地网卡到远端网卡连接 | 如果网卡需要 QP/session/channel，也抽 `CustomEndpoint` |
| metadata 暴露远端 key | 定制后端必须定义可序列化的 remote access token |
| worker pool 解耦提交和完成 | 有异步 completion queue 时应照 RDMA 做 |
| retry/redispatch 与 topology 绑定 | 多网卡场景要定义失败后的替代路径选择 |

【图示：RDMA 后端切片、建连与完成路径】

## 6. 新增定制网卡后端的最小路径

假设新协议名为 `customnic`。

### 6.1 最小文件结构

```
mooncake-transfer-engine/include/transport/customnic_transport/customnic_transport.h
mooncake-transfer-engine/src/transport/customnic_transport/customnic_transport.cpp
mooncake-transfer-engine/src/transport/customnic_transport/CMakeLists.txt
```

如果网卡有异步队列，建议进一步拆：

```
customnic_context.{h,cpp}
customnic_endpoint.{h,cpp}
customnic_worker_pool.{h,cpp}
```

### 6.2 接入编译和工厂

1. 在顶层 CMake 增加 `USE_CUSTOMNIC` 选项。
2. 在 `src/transport/CMakeLists.txt` 纳入 `customnic_transport` object library。
3. 在 [multi_transport.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:305) include 头文件，并加入：

```cpp
#ifdef USE_CUSTOMNIC
else if (std::string(proto) == "customnic") {
    transport = new CustomNicTransport();
}
#endif
```

### 6.3 定义 metadata 契约

最简做法是复用 `BufferDesc` 的通用字段：

| 信息 | 可放位置 | 说明 |
|---|---|---|
| 本地地址 | `BufferDesc::addr` | 必须 |
| 长度 | `BufferDesc::length` | 必须 |
| remote key / token | 可新增字段，或临时复用 `rkey` vector | 正式实现建议新增清晰字段 |
| device 列表 | `SegmentDesc::devices` | 多网卡必需 |
| topology | `SegmentDesc::topology` | 需要亲和性/多路径时必需 |
| 数据端口或 endpoint address | `RpcMetaDesc` 或新增字段 | 如果不是 RDMA handshake，需要显式发布 |

如果只是先跑通功能，可仿 TCP：segment `protocol = "customnic"`，buffer 只发布 `addr/length`，另开自定义数据端口。但如果目标是定制网卡零拷贝，建议仿 RDMA：注册内存得到 per-device key，把 key 放进 metadata。

### 6.4 实现顺序

1. **骨架后端**：继承 `Transport`，实现 install/register/unregister/submit/status。
2. **单连接、单 slice**：先不切片、不多网卡，像 TCP 一样每个 request 一个 slice。
3. **完成语义打通**：底层回调或 polling 完成后只做两件事：`slice->markSuccess()` 或 `slice->markFailed()`。
4. **remote key 支持**：注册内存时写入定制 key，发起端从目标 segment 读取 key。
5. **多网卡与 topology**：引入 `CustomContext`，按 location 选择本地/远端 device。
6. **重试和失败隔离**：参考 RDMA worker pool，失败时换 endpoint/device。

### 6.5 新后端最小伪代码

```cpp
class CustomNicTransport : public Transport {
 public:
  int install(std::string& local, std::shared_ptr<TransferMetadata> meta,
              std::shared_ptr<Topology> topo) override {
    metadata_ = meta;
    local_server_name_ = local;
    initDeviceOrServer();
    publishSegment("customnic");
    return metadata_->updateLocalSegmentDesc();
  }

  int registerLocalMemory(void* addr, size_t len, const std::string& loc,
                          bool remote_accessible, bool update_metadata) override {
    auto key = customnic_register(addr, len);
    BufferDesc desc;
    desc.name = loc;
    desc.addr = reinterpret_cast<uint64_t>(addr);
    desc.length = len;
    desc.rkey = {key};  // 原型阶段可复用；正式建议新增字段。
    return metadata_->addLocalMemoryBuffer(desc, update_metadata);
  }

  Status submitTransferTask(const std::vector<TransferTask*>& tasks) override {
    for (auto* task : tasks) {
      auto& req = *task->request;
      Slice* s = getSliceCache().allocate();
      s->source_addr = req.source;
      s->length = req.length;
      s->opcode = req.opcode;
      s->target_id = req.target_id;
      s->status = Slice::PENDING;
      task->slice_list.push_back(s);
      __sync_fetch_and_add(&task->slice_count, 1);
      submitToCustomNic(s, req.target_offset);
    }
    return Status::OK();
  }
};
```

## 7. 最简测试方案

### 7.1 编译级测试

先只要求后端能编译进工厂：

```bash
cmake -S deps/Mooncake -B build-mooncake-customnic \
  -DUSE_TCP=ON -DUSE_CUSTOMNIC=ON
cmake --build build-mooncake-customnic --target transfer_engine_validator -j
```

### 7.2 TCP 基线功能测试

TCP 不依赖 RDMA 硬件，适合作为环境 sanity check。`transfer_engine_validator` 已经包含 target/initiator 双角色、内存注册、openSegment、WRITE 后 READ 回来并 `memcmp` 校验数据，见 [transfer_engine_validator.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/example/transfer_engine_validator.cpp:394)。

两个终端：

```bash
./build-mooncake-customnic/mooncake-transfer-engine/example/transfer_engine_validator \
  --mode=target \
  --protocol=tcp \
  --metadata_server=P2PHANDSHAKE \
  --local_server_name=127.0.0.1:12345 \
  --buffer_size=16777216 --block_size=4096 --batch_size=4 --threads=1
```

```bash
./build-mooncake-customnic/mooncake-transfer-engine/example/transfer_engine_validator \
  --mode=initiator \
  --protocol=tcp \
  --metadata_server=P2PHANDSHAKE \
  --local_server_name=127.0.0.1:12346 \
  --segment_id=127.0.0.1:12345 \
  --buffer_size=16777216 --block_size=4096 --batch_size=4 --threads=1 --duration=3
```

如果 `P2PHANDSHAKE` 触发动态端口映射，按 target 日志打印的实际 server name/port 调整 `--segment_id`。

### 7.3 定制后端最小功能测试

把上面 `--protocol=tcp` 换成 `--protocol=customnic`。第一阶段成功标准：

1. target 进程 install 成功并发布 `protocol=customnic`。
2. initiator `openSegment()` 能拿到目标 segment。
3. `registerLocalMemory()` 后 metadata 里能看到 buffer。
4. WRITE + READ 回环通过 validator 的数据一致性检查。

### 7.4 RDMA 对照测试

有 RDMA 设备时，用同一 validator 做对照：

```bash
./transfer_engine_validator --mode=target --protocol=rdma \
  --metadata_server=P2PHANDSHAKE --local_server_name=<target_ip:port> \
  --device_name=mlx5_0 --buffer_size=16777216 --block_size=4096 --batch_size=4 --threads=1
```

```bash
./transfer_engine_validator --mode=initiator --protocol=rdma \
  --metadata_server=P2PHANDSHAKE --local_server_name=<init_ip:port> \
  --segment_id=<target_ip:port> --device_name=mlx5_0 \
  --buffer_size=16777216 --block_size=4096 --batch_size=4 --threads=1 --duration=3
```

如果手动传 `--device_name`，validator 会生成 NIC priority matrix 并通过 `installTransport("rdma", args)` 传入，见 [transfer_engine_validator.cpp](/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/mooncake-transfer-engine/example/transfer_engine_validator.cpp:367)。

### 7.5 单元测试建议

新增后端后，至少补三类测试：

| 测试 | 方法 | 价值 |
|---|---|---|
| 工厂测试 | `installTransport("customnic") != nullptr` | 防止 CMake/宏/工厂漏接 |
| metadata 测试 | 注册后检查 `SegmentDesc.protocol/buffers/key` | 防止远端无法寻址 |
| 状态机测试 | mock completion 调用 success/fail | 防止 batch 永远 WAITING |

## 8. 理解验证与质量检查

### 理解验证状态

| 核心概念 | 自我解释 | 理解 WHY | 应用迁移 | 状态 |
|---|---|---|---|---|
| `Transport` 抽象 | 已说明接口和 slice/task | 已说明屏蔽后端差异 | 可迁移到 customnic | 完成 |
| `MultiTransport` 路由 | 已说明按 segment protocol | 已说明元数据驱动 | 可新增 proto 分支 | 完成 |
| TCP 后端 | 已说明端口、会话、数据流 | 已说明为何最简单 | 可作为原型模板 | 完成 |
| RDMA 后端 | 已说明 MR、key、endpoint、CQ | 已说明高性能路径原因 | 可借鉴多网卡设计 | 完成 |
| 最小测试 | 已给 validator 命令 | 已说明先 TCP 再 customnic | 可直接执行 | 完成 |

### 质量检查清单

- [x] 章节结构完整，由浅入深
- [x] Markdown 已标注图示位置，HTML 已内联 SVG 图示
- [x] TCP 和 RDMA 均有源码级分析
- [x] 给出新增后端最小实现路径
- [x] 给出最简测试方案
- [x] 元信息已标注
- [ ] IMA 知识库同步：当前会话未暴露 `ima-skills` 工具，未执行
- [ ] 腾讯文档同步：当前会话未暴露 `tencent-docs` MCP 工具，未执行

## 9. 实验验证结果

验证日期：2026-05-22。构建目录：`/shared/vllm-xcpu-dev-kit-0.19/deps/Mooncake/build`。

### 9.1 测试环境说明

我们在一台常规 x86 服务器环境中进行了 TCP 相关实验。该环境没有可用 RDMA 后端：

```bash
$ ls /sys/class/infiniband
# 无输出
$ ibv_devices
/bin/bash: line 1: ibv_devices: command not found
```

因此 RDMA 端到端测试暂不可用。后续有 RDMA 环境时，需要安装/暴露 `rdma-core` 工具和 HCA 设备，再跑 `--protocol=rdma --device_name=<mlx5_x>`。

此外，该实验环境缺少 NUMA 迁页/绑定权限，`memory_location_test` 不适合作为该环境的可用测试：

```text
mbind: Operation not permitted
numa_move_pages failed ... Operation not permitted
```

旧版 `tcp_transport_test` 和 `transfer_metadata_test` 默认依赖 `MC_METADATA_SERVER`。若直接用空值会走 etcd 并 fatal；若强行用 `P2PHANDSHAKE`，测试内硬编码的 `127.0.0.2:12345` 会和 P2P 动态端口不一致。因此本次使用一个内存型 HTTP metadata server，等价实现 `GET/PUT/DELETE /metadata?key=...`，并设置：

```bash
export MC_METADATA_SERVER=http://127.0.0.1:18080/metadata
```

### 9.2 Transfer Engine 非 RDMA 单测

命令：

```bash
ctest --output-on-failure -R '^(default_config_test|transport_uint_test|endpoint_store_test|endpoint_store_integration_test|tcp_transport_test|transfer_metadata_test|topology_test|common_test)$'
```

结果节选：

```text
1/8 Test #1: default_config_test ...............   Passed
2/8 Test #2: transport_uint_test ...............   Passed
3/8 Test #3: endpoint_store_test ...............   Passed
4/8 Test #4: endpoint_store_integration_test ...   Passed
5/8 Test #5: tcp_transport_test ................   Passed
6/8 Test #6: transfer_metadata_test ............   Passed
7/8 Test #7: topology_test .....................   Passed
8/8 Test #9: common_test .......................   Passed

100% tests passed, 0 tests failed out of 8
Total Test time (real) = 4.15 sec
```

`transfer_metadata_test` 单独结果节选：

```text
[ RUN      ] TransferMetadataTest.LocalSegmentTest
[       OK ] TransferMetadataTest.LocalSegmentTest
[ RUN      ] TransferMetadataTest.LocalMemoryBufferTest
[       OK ] TransferMetadataTest.LocalMemoryBufferTest
[ RUN      ] TransferMetadataTest.RpcMetaEntryTest
[       OK ] TransferMetadataTest.RpcMetaEntryTest
[  PASSED  ] 3 tests.
```

### 9.3 TCP 端到端数据一致性测试

命令：

```bash
./mooncake-transfer-engine/example/transfer_engine_validator \
  --mode=target --protocol=tcp \
  --metadata_server=http://127.0.0.1:18080/metadata \
  --local_server_name=127.0.0.1:19001 \
  --buffer_size=16777216 --block_size=4096 --batch_size=4 --threads=1

./mooncake-transfer-engine/example/transfer_engine_validator \
  --mode=initiator --protocol=tcp \
  --metadata_server=http://127.0.0.1:18080/metadata \
  --local_server_name=127.0.0.1:19002 \
  --segment_id=127.0.0.1:19001 \
  --buffer_size=16777216 --block_size=4096 --batch_size=4 --threads=1 \
  --duration=3
```

结果节选：

```text
Worker 0 stopped! Data validation passed
Test completed: duration 3.00, batch count 2080, throughput 0.01 GB/s
```

这说明 TCP 后端的 install、metadata 发布、内存注册、openSegment、WRITE、READ、完成状态和数据一致性路径均可用。

### 9.4 TCP benchmark smoke test

命令：

```bash
./mooncake-transfer-engine/example/transfer_engine_bench \
  --mode=target --operation=write --protocol=tcp \
  --metadata_server=http://127.0.0.1:18080/metadata \
  --local_server_name=127.0.0.1:19011 \
  --buffer_size=16777216 --block_size=4096 --batch_size=4 --threads=1

./mooncake-transfer-engine/example/transfer_engine_bench \
  --mode=initiator --operation=write --protocol=tcp \
  --metadata_server=http://127.0.0.1:18080/metadata \
  --local_server_name=127.0.0.1:19012 \
  --segment_id=127.0.0.1:19011 \
  --buffer_size=16777216 --block_size=4096 --batch_size=4 --threads=1 \
  --duration=3
```

结果节选：

```text
Worker 0 stopped!
Test completed: duration 3.00, batch count 3939, throughput 0.02 GB/s
```

benchmark 只作为吞吐 smoke test；功能正确性以 validator 的数据校验为准。

## 10. 一页总结

新增 Mooncake 网络后端的关键是接入四层契约：

1. **工厂契约**：在 `MultiTransport::installTransport()` 注册 `proto -> Transport`。
2. **metadata 契约**：本地 segment 必须发布 `protocol`，buffer 必须发布远端访问需要的 addr/length/key/device 信息。
3. **任务契约**：`submitTransferTask()` 必须把 `TransferTask` 拆成 slice，提交到底层网卡系统。
4. **完成契约**：底层完成必须最终映射成 `slice->markSuccess()` 或 `slice->markFailed()`，否则 `getTransferStatus()` 永远不会完成。

实现策略建议：第一版按 TCP 做单连接、单 slice、数据一致性通过；第二版加入定制网卡 memory registration 和 remote key；第三版再按 RDMA 引入 context/endpoint/worker pool/topology/retry。
