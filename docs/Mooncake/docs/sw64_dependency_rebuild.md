# SW64 依赖重编与 Mooncake 构建脚本

前置依赖项: 编译了动态库的 python3，假定 python3 安装 /usr/local/ 底下。如果不同，需要同步修改 sw64_rebuild_deps.sh 偏后的 Mooncake cmake 选项。

```
export LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:$LIBRARY_PATH  
export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH
export LDFLAGS="$LDFLAGS -latomic"
```

本文档对应脚本：

```bash
scripts/sw64_rebuild_deps.sh
```

脚本用于国产 SW64 节点上的 Mooncake 依赖重编。它不会把依赖安装到
`/usr`、`/usr/local` 等系统路径，而是统一安装到 Mooncake 仓库内的私有前缀：

```bash
Mooncake/deps/_install
```

## 版本基线

本地 x86 成功编译 Mooncake 时使用的依赖版本如下，脚本默认按这些版本对齐：

| 依赖 | 版本 / ref |
| --- | --- |
| gflags | `v2.2.2` |
| glog | `v0.6.0` |
| yalantinglibs | `6a0e067d9a43492cf8e4e280b531924fbd724dbd`，即 `0.6.0-6-g6a0e067` |
| yaml-cpp | `0.8.0` |
| msgpack-cxx | x86 包版本为 `6.1.0`，源码构建使用 `cpp-6.1.0` |
| Python | `3.12` |

其中 `glog v0.6.0` 很关键。较新的 glog 移除了
`LogSink::send(..., const std::tm*, ...)` 兼容重载，当前 Mooncake 测试代码会因此出现
`override` 编译错误。

## 系统包

系统包仍需要由系统包管理器提供，但源码依赖不会安装到系统路径：

```bash
sudo yum install -y \
    cmake \
    gcc gcc-c++ make git \
    libibverbs-devel \
    numactl numactl-devel \
    gtest gtest-devel \
    gmock gmock-devel \
    boost-devel \
    openssl-devel \
    hiredis hiredis-devel \
    libcurl libcurl-devel \
    zstd zstd-devel \
    xxhash xxhash-devel \
    jsoncpp jsoncpp-devel
```

## 脚本行为

默认行为：

- 从 `MOONCAKE_ROOT` 找到 Mooncake 根目录，默认是
  `/home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake`。
- 依赖源码放在 `Mooncake/deps`。
- 依赖私有安装路径为 `Mooncake/deps/_install`。
- 仓库已存在时不会重新 clone；如果本地已有目标 ref，就直接
  `git checkout <ref>`，只有缺少目标 ref 时才执行 `git fetch --tags origin`。
- 仓库不存在时才执行 `git clone`。
- 默认清理旧构建目录，保证干净环境：
  - `Mooncake/build`
  - `Mooncake/deps/_install`
  - `Mooncake/deps/*/build`
  - `Mooncake/deps/*/build-sw64`
- 所有编译使用最大并行度：`JOBS=$(nproc)`。
- 默认记录完整日志到 `Mooncake/build/sw64_rebuild_deps.log`。如不需要日志，可用
  `LOG_FILE=` 关闭；也可以用 `LOG_FILE=/path/to/log` 指定位置。
- 依赖构建完成后会重新配置 Mooncake，并默认执行 `cmake --build build -j$(nproc)`。

## 重入性说明

脚本按两种方式支持再入：

- 默认 `CLEAN=1`：适合中途失败后重新执行。脚本会删除 `Mooncake/build`、
  `deps/_install`、各依赖的 `build` 和 `build-sw64`，然后从源码 checkout 和
  CMake configure 阶段重新开始。这是最可靠的再入方式。
- `CLEAN=0`：适合确认环境无误后的增量重跑。CMake 会复用已有 build 目录和
  `_install`，速度更快，但如果上一次失败发生在 `cmake --install` 中途，私有安装前缀
  可能处于半更新状态；此时建议使用默认 `CLEAN=1`。

脚本不会删除依赖源码仓库。仓库已存在时，如果本地已有目标 ref，就直接 checkout；
只有本地缺失目标 ref 时才 fetch。若源码仓库内存在未提交改动导致 checkout 失败，
脚本会立即退出，避免覆盖人工修改。

## 环境变量

脚本会自动设置：

```bash
export PATH="${DEPS_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${DEPS_PREFIX}/lib64:${DEPS_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${DEPS_PREFIX}/lib64:${DEPS_PREFIX}/lib:${LIBRARY_PATH:-}"
export CPATH="${DEPS_PREFIX}/include:${CPATH:-}"
export PKG_CONFIG_PATH="${DEPS_PREFIX}/lib64/pkgconfig:${DEPS_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${DEPS_PREFIX}:${CMAKE_PREFIX_PATH:-}"
```

脚本还会生成运行期环境文件：

```bash
Mooncake/deps/env.sh
```

后续运行 Mooncake 产物或测试程序前，需要在当前 shell 中加载它，否则动态链接器可能找不到
`deps/_install/lib` 或 `deps/_install/lib64` 下的 `.so`，例如 `libyaml-cpp.so.0.8`：

```bash
cd /home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake
source deps/env.sh
```

Mooncake 配置时还会显式指定：

```bash
-Dgflags_DIR=<deps/_install 中的 gflags-config.cmake 所在目录>
-Dglog_DIR=<deps/_install 中的 glog-config.cmake 所在目录>
-Dyalantinglibs_DIR=<deps/_install 中的 yalantinglibsConfig.cmake 所在目录>
-Dyaml-cpp_DIR=<deps/_install 中的 yaml-cpp-config.cmake 所在目录>
-DCMAKE_PREFIX_PATH=<Mooncake/deps/_install>
```

这样可以避免 CMake 误用 `/usr/local` 或系统路径中的新版 glog/yaml-cpp。

## 执行方式

在远端节点执行：

```bash
source /home/wuzhikun/vllm-xcpu-dev-kit-0.19/.venv/bin/activate
cd /home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake
bash scripts/sw64_rebuild_deps.sh
```

常用覆盖项：

```bash
# 只重编依赖并配置 Mooncake，不编译 Mooncake 主工程
BUILD_MOONCAKE=0 bash scripts/sw64_rebuild_deps.sh

# 保留已有 build 目录和 deps/_install
CLEAN=0 bash scripts/sw64_rebuild_deps.sh

# 指定并行度
JOBS=64 bash scripts/sw64_rebuild_deps.sh
```

## 运行测试前的动态库路径

因为依赖没有安装到系统路径，直接运行 build 目录下的二进制可能报错：

```text
error while loading shared libraries: libyaml-cpp.so.0.8: cannot open shared object file: No such file or directory
```

运行前加载脚本生成的环境文件。这个端到端测试必须使用 3 个终端：

| 终端 | 作用 | 预期执行时间 |
| --- | --- | --- |
| 终端 1 | HTTP metadata server，保存 target 发布的 segment/rpc metadata | 常驻；整个测试期间不要退出，测试结束后按 Ctrl-C 停止 |
| 终端 2 | target，注册本地 buffer 并等待 initiator 连接和传输 | 常驻；启动后看起来会停在前台，这是正常等待；initiator 结束后再按 Ctrl-C 停止 |
| 终端 3 | initiator，打开 target segment，执行 WRITE/READ 并校验数据 | 约 `--duration=3` 秒，加少量初始化时间；看到 `Data validation passed` 后自动退出 |

这里使用 `scripts/http_metadata_server.py` 是为了让 SW64/本地 smoke test 在没有额外
Python 包和 Go module 下载的情况下可复现。它只用 Python 标准库实现测试所需的
`GET/PUT/DELETE /metadata?key=...`，保存 Transfer Engine 的 segment、rpc meta 和
buffer metadata。数据传输本身不经过这个脚本；它只服务控制面发现流程。

更正式的测试建议使用以下 metadata 服务之一：

- `etcd`：生产/集群测试优先选项。依赖 `etcd` 服务端和客户端访问端口，默认常见端口是
  `2379`；target 和 initiator 使用同一个 `--metadata_server=<etcd_host>:2379`。
- Mooncake 自带 Go HTTP metadata server：位于
  `mooncake-transfer-engine/example/http-metadata-server`，依赖 Go toolchain 和 Go module
  下载，例如 `github.com/gin-gonic/gin`。
- Mooncake 自带 Python HTTP metadata server：位于
  `mooncake-transfer-engine/example/http-metadata-server-python/bootstrap_server.py`，依赖
  `aiohttp`。

正式性能测试不建议使用 `scripts/http_metadata_server.py` 评估控制面性能；它只用于本文档
的单机端到端连通性和数据一致性验证。

```bash
cd /home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake
source deps/env.sh

# 终端 1: HTTP metadata server，常驻到整个测试结束。
python3 scripts/http_metadata_server.py --host 127.0.0.1 --port 18080
```

```bash
cd /home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake
source deps/env.sh

# 终端 2: target，常驻等待 initiator，不要在 initiator 结束前按 Ctrl-C。
./build/mooncake-transfer-engine/example/transfer_engine_validator \
    --mode=target \
    --protocol=tcp \
    --metadata_server=http://127.0.0.1:18080/metadata \
    --local_server_name=127.0.0.1:12345 \
    --buffer_size=16777216 \
    --block_size=4096 \
    --batch_size=4 \
    --threads=1
```

```bash
cd /home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake
source deps/env.sh

# 终端 3: initiator，运行约 3 秒后自动退出。
# 看到 "Data validation passed" 表示 TCP 端到端数据校验通过。
./build/mooncake-transfer-engine/example/transfer_engine_validator \
    --mode=initiator \
    --protocol=tcp \
    --metadata_server=http://127.0.0.1:18080/metadata \
    --local_server_name=127.0.0.1:12346 \
    --segment_id=127.0.0.1:12345 \
    --buffer_size=16777216 \
    --block_size=4096 \
    --batch_size=4 \
    --threads=1 \
    --duration=3
```

也可以只对单次命令设置动态库路径。仍建议用 HTTP metadata server 固定 target segment；
如果改用 `P2PHANDSHAKE`，initiator 的 `--segment_id` 需要填 target 日志里
`RPC using P2P handshake, listening on ...` 打印的动态 RPC 地址，而不是
`--local_server_name`：

```bash
LD_LIBRARY_PATH=/home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake/deps/_install/lib64:/home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake/deps/_install/lib:${LD_LIBRARY_PATH:-} \
./build/mooncake-transfer-engine/example/transfer_engine_validator \
    --mode=target \
    --protocol=tcp \
    --metadata_server=http://127.0.0.1:18080/metadata \
    --local_server_name=127.0.0.1:12345 \
    --buffer_size=16777216 \
    --block_size=4096 \
    --batch_size=4 \
    --threads=1
```

## 实测预期结果

本次脚本执行后应至少满足：

- `deps/_install` 下存在以下 CMake package 文件：
  - `gflags-config.cmake`
  - `glog-config.cmake`
  - `yalantinglibsConfig.cmake`
  - `yaml-cpp-config.cmake`
  - `msgpack` 或 `msgpack-cxx` 相关 CMake 配置文件
- `build/CMakeCache.txt` 中的依赖路径应指向 `Mooncake/deps/_install`，而不是
  `/usr/local`。
- `glog_DIR` 应指向私有安装的 `glog v0.6.0`。
- 运行测试程序前执行 `source deps/env.sh` 后，动态链接器应能找到
  `libyaml-cpp.so.0.8`、`libglog.so.1`、`libgflags.so.2.2` 等私有依赖库。
- 之前的 `ClientIdCaptureSink::send(...) marked override, but does not override`
  错误不应再出现。

## 本次实测关键过程

远端节点：`sw8a-12`。

已观察到的关键过程：

- 脚本以默认 `CLEAN=1` 启动，清理了旧的 `Mooncake/build`、`deps/_install` 和依赖
  `build-sw64` 目录。
- `gflags v2.2.2` 使用 `-j256` 成功编译并安装到 `deps/_install`；在你修复
  `/home/wuzhikun/.cmake/` 权限后，gflags 写入 CMake package registry 正常。
- `glog v0.6.0` 使用 `deps/_install` 中的 gflags 成功配置、编译并安装到
  `deps/_install/lib64/cmake/glog`。
- `yalantinglibs 6a0e067d...` 成功配置并安装到 `deps/_install`。
- `yaml-cpp` 的实际 tag 名是 `0.8.0`，不是 `yaml-cpp-0.8.0`；脚本已按实测修正。
- Mooncake 主工程配置后，编译命令中出现
  `-isystem /home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake/deps/_install/include`，
  表明私有依赖路径已生效。
- Mooncake 当前以 `cmake --build build -j256` 编译，已经进入 `mooncake_store`
  阶段；之前的 glog `LogSink::send` override 错误未再出现。

最终完成或失败结果会继续补充到本节。
