from mpi4py import MPI
import numpy as np

def main():
    # 初始化通信器
    comm = MPI.COMM_WORLD
    # 获取当前进程的秩 (Rank)
    rank = comm.Get_rank()
    # 获取总进程数 (Size)
    size = comm.Get_size()

    # --- 1. 基础信息打印 ---
    print(f"Hello! 我是第 {rank} 号进程，总共有 {size} 个进程。")

    # 等待所有进程同步
    comm.Barrier()

    # --- 2. 点对点通信 (Point-to-Point) ---
    # 进程 0 发送数据给进程 1
    if rank == 0:
        data = {'key': 'value', 'number': 42}
        if size > 1:
            comm.send(data, dest=1, tag=11)
            print(f"进程 {rank}: 已发送数据到进程 1")
    elif rank == 1:
        data = comm.recv(source=0, tag=11)
        print(f"进程 {rank}: 收到来自进程 0 的数据: {data}")

    # --- 3. 集体通信 (Collective Communication: Reduction) ---
    # 每个进程生成一个随机数，然后求和
    local_val = np.array([float(rank)], dtype='d')
    sum_val = np.array([0.0], dtype='d')

    # 将所有进程的 local_val 相加，结果存入 root (进程 0) 的 sum_val
    comm.Reduce(local_val, sum_val, op=MPI.SUM, root=0)

    if rank == 0:
        # 验证结果：0 + 1 + ... + (size-1) = size * (size-1) / 2
        expected = size * (size - 1) / 2
        print(f"--- 验证 ---")
        print(f"所有进程 Rank 的求和结果为: {sum_val[0]} (预期值: {expected})")

if __name__ == "__main__":
    main()
