#include <iostream>
#include <vector>
#include <mpi.h>

int main(int argc, char** argv) {
    // 1. 初始化 MPI 环境
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // 获取总进程数

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // 获取当前进程 ID

    // 获取运行该进程的机器名
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    std::cout << "Hello! 我是来自 " << processor_name 
              << " 的进程 " << world_rank << "/" << world_size << std::endl;

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    // --- 2. 点对点通信 (Point-to-Point) ---
    if (world_rank == 0) {
        int send_num = 100;
        if (world_size > 1) {
            // 参数：数据地址, 计数, 类型, 目标, 标签, 通信器
            MPI_Send(&send_num, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            std::cout << "进程 0: 已向进程 1 发送数字 " << send_num << std::endl;
        }
    } else if (world_rank == 1) {
        int recv_num = 0;
        MPI_Recv(&recv_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "进程 1: 已收到来自进程 0 的数字 " << recv_num << std::endl;
    }

    // --- 3. 集体通信 (Collective: Reduce) ---
    int local_val = world_rank;
    int global_sum = 0;

    // 将所有进程的 rank 值相加
    // 参数：发送缓冲区, 接收缓冲区, 计数, 类型, 操作, 根进程, 通信器
    MPI_Reduce(&local_val, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        int expected = (world_size * (world_size - 1)) / 2;
        std::cout << "--- 验证 ---" << std::endl;
        std::cout << "所有进程 Rank 求和结果: " << global_sum 
                  << " (预期: " << expected << ")" << std::endl;
    }

    // 4. 清理环境
    MPI_Finalize();
    return 0;
}
