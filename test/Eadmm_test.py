import ast
import time
import math

import numpy as np

from load_data import load_data
import torch
import matplotlib.pyplot as plt
import json

AUTO_STOP = True
GPU = False  # GPU save figure

# 定义 ADMM 主过程
# 输入：RHO 是 ADMM 的罚参数 ρ，iter 是最大迭代次数
# 输出：None
def eadmm(RHO, iter):
    dataset = load_data()
    customer_names = dataset['customer_names']
    site_names = dataset['site_names']
    demand = dataset['demand']
    is_link = dataset['is_link']
    capacity = dataset['capacity']
    time_list = dataset['time_list']

    # parameters
    big_num = int(1e9)
    I = min(customer_names.size, big_num)
    J = min(site_names.size, big_num)
    # 使用第一天的数据（288个时间点）
    T = min(demand.shape[0], 288)  # 限制为第一天的数据
    print(f"使用第一天数据: {T}个时间点")
    customer_num = I
    site_num = J

    # 每个站点的最大免费时段数 - 根据完整数据调整
    L = math.ceil(T * 0.05)  # 5%的时间可以免费
    print(f"每个站点最大免费时段数: {L}")
    
    max_iter = iter
    rho = RHO
    # 保存目标值轨迹到不同ρ的文件里，便于对比
    filename = 'experiment_results/RHO='+str(RHO)+'.txt'

    # A 矩阵：客户到站点的流量关系
    # 如果 is_link[j, i] = 1，则表示站点 j 和客户 i 之间是连接的，A[j, i * site_num + j] = 1 即建立了对应的流量关系。
    A = torch.zeros((J, I * J), dtype=torch.float32, requires_grad=False)
    for j in range(site_num):
        for i in range(customer_num):
            A[j, i * site_num + j] = int(is_link[j, i])

    # B 矩阵：站点到客户的流量关系
    # 如果 is_link[j, i] = 1，则表示站点 j 和客户 i 之间是连接的，B[i, i * site_num + j] = 1 即建立了对应的流量关系。
    B = torch.zeros((I, I * J), dtype=torch.float32, requires_grad=False)
    for i in range(customer_num):
        for j in range(site_num):
            B[i, i * site_num + j] = int(is_link[j, i])

    # C 矩阵：站点的带宽容量
    # C 是一个大小为 J 的张量，表示每个站点的带宽容量。
    C = torch.from_numpy(capacity[:J]/1000).to(torch.float32)
    C.requires_grad = False

    # D 矩阵：客户的需求
    # D 是一个大小为 T x I 的张量，表示每个时间点 t 每个客户 i 的需求。
    D = torch.from_numpy(demand[:T, :I]/1000).to(torch.float32)
    D.requires_grad = False

    # y_k 是一个大小为 T x J 的张量，表示每个时间点 t 每个站点 j 是否免费/计费时隙
    y_k = torch.zeros((T, J), dtype=torch.float32, requires_grad=False)
    # print('numpy version :  ', np.__version__)
    # free_slots_str = str(np.loadtxt('./test_solution/free_slots_M.txt', dtype=str, delimiter="\n"))
    # free_slots_str = str(np.loadtxt('./test_solution/free_slots_M.txt', dtype=str, delimiter=";")[0])
    free_slots_str = str(np.loadtxt('./test_solution/free_slots_M8928.txt', dtype=str, delimiter=";"))
    free_slots_dic = ast.literal_eval(free_slots_str)
    for t in range(T):
        for site_index in free_slots_dic[t]:
            y_k[t, site_index] = 1

    AT = A.T  # (I*J)*J
    AT.requires_grad = False
    BT = B.T  # (I*J) * I
    BT.requires_grad = False
    ATA = torch.matmul(AT, A) # (I*J) * (I*J)
    ATA.requires_grad = False
    BTB = torch.matmul(BT, B) # (I*J) * (I*J)
    BTB.requires_grad = False
    diag = torch.eye(I * J) # (I*J) * (I*J)的单位矩阵
    SUM = ATA + BTB + diag  # reverse精度问题 + 1e-6* torch.eye(I*J)
    INV = torch.linalg.inv(SUM) # 求出SUM的逆矩阵
    INV.requires_grad = False

    # ATB = torch.matmul(AT, (y_k - 1).T)

    # initialization
    # x_k1 是一个大小为 (I*J) x T 的张量，表示每个时间点 t 每个客户 i 到每个站点 j 的流量
    x_k1 = torch.zeros((I * J, T), dtype=torch.float32, requires_grad=False)
    # l1_k1, l2_k1, l3_k1 是 ADMM 算法中的三个 Lagrange  multiplier 变量，分别对应于流量约束、需求约束和容量约束。
    l1_k1 = torch.zeros((T, J), dtype=torch.float32, requires_grad=False)
    l2_k1 = torch.zeros((T, I), dtype=torch.float32, requires_grad=False)
    l3_k1 = torch.zeros((T, I * J), dtype=torch.float32, requires_grad=False)
    # s_k1,s_k2是松弛变量
    s_k2 = torch.zeros((T, I * J), dtype=torch.float32, requires_grad=False)
    s_k1 = torch.zeros((T, J), dtype=torch.float32, requires_grad=False)
    # b_k1 是一个大小为 J 的张量，表示每个站点的计费带宽
    b_k1 = torch.zeros(J, dtype=torch.float32, requires_grad=False)

    # x_k1 = torch.zeros((I * J, T), dtype=torch.float32, requires_grad=False)
    #
    # l1_k = torch.zeros((T, J), dtype=torch.float32, requires_grad=False)
    # l2_k = torch.zeros((T, I), dtype=torch.float32, requires_grad=False)
    # l3_k = torch.zeros((T, I * J), dtype=torch.float32, requires_grad=False)
    # s2_k = torch.zeros((T, I * J), dtype=torch.float32, requires_grad=False)
    # s_k = torch.zeros((T, J), dtype=torch.float32, requires_grad=False)
    # b_k = torch.zeros(J, dtype=torch.float32, requires_grad=False)


    # GPU
    # if torch.backends.mps.is_available():
    # device = torch.device("mps")  # apple M chip
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # if torch.cuda.is_available():
    # device = torch.device("cuda:0")
    if torch.backends.mps.is_available():
        print('mps available')
        # GPU = True
        device = torch.device('mps') # apple M chip
        print('use gpu M4 Max')
        # device = torch.device('cpu')
        # print('use cpu')

        A = A.to(device)
        B = B.to(device)
        C = C.to(device)
        D = D.to(device)
        INV = INV.to(device)
        AT = AT.to(device)
        BT = BT.to(device)

        y_k = y_k.to(device)
        l1_k1 = l1_k1.to(device)
        l2_k1 = l2_k1.to(device)
        l3_k1 = l3_k1.to(device)
        s_k1 = s_k1.to(device)
        s_k2 = s_k2.to(device)
        b_k1 = b_k1.to(device)
        x_k1 = x_k1.to(device)

    # start ADMM
    lst_obj = [] # 储存每次的目标函数值（最小化计费带宽b_k1总和），用于绘制收敛曲线
    ratio = []
    square_error = []
    sum_x = 0 # 中间变量，用于计算和更新其他变量
    sum_x2 = 0 # 中间变量，用于计算和更新其他变量
    lst_rho = [] # 储存每次的rho值

    start = time.time()
    with torch.no_grad(): # 禁用梯度计算，因为我们只需要进行前向传播
        for iteration in range(max_iter):

            # if iteration == 0:
            #     common_term = sum_x - (C - b_k1) * y_k - b_k1

            # 计算所有的平方项
            # square_error1 = torch.sum(torch.square(common_term + s_k1))
            # square_error2 = torch.sum(torch.square(sum_x2 - D))
            # square_error3 = torch.sum(torch.square(x_k1.T - s_k2))

            # ratio.append((torch.sum(square_error1) + torch.sum(square_error2) + square_error3) / (
            #         torch.sum(b_k1) + torch.sum(torch.abs(l2_k1 * (sum_x2 - D))) + torch.sum(
            #     l1_k1 * (torch.abs(common_term + s_k1)) + torch.sum(l3_k1 * (x_k1.T - s_k1)))))

            # x 更新：ADMM 中，x 变量代表从客户到站点的流量。每次迭代中，x 通过解线性方程组来更新。
            const_c = ((C - b_k1) * y_k + b_k1 - s_k1).to(torch.float32)  # T*J
            atc = torch.matmul(AT, const_c.T)  # (I*J)*T
            btd = torch.matmul(BT, D.T)  # (I*J)*T
            atl1 = torch.matmul(AT, l1_k1.T)  # (I*J)*T
            atl2 = torch.matmul(BT, l2_k1.T)  # (I*J)*T
            equ_right = ((atc + btd + s_k2.T) - 1.0 / rho * (atl1 + atl2 + l3_k1.T))

            x_k1[:] = (torch.matmul(INV, equ_right))
            # 计算站点总流量，用于后续验证可行性
            A1X = torch.matmul(A, x_k1)

            # b 更新：更新站点的带宽 b。这是一个闭式解，可以通过简单的线性计算获得
            down = rho * torch.sum(torch.square(y_k - 1), dim=0)
            sum_x = torch.matmul(A, x_k1).T  # T*J
            up = torch.sum(rho * (y_k - 1) * (C * y_k - s_k1 - sum_x - l1_k1/rho), dim=0) - 1
            b_k1[:] = (up / down).clip(0, torch.inf)

            # s 更新：ADMM 中的松弛变量更新，确保不等式约束满足
            common_term = sum_x - (C - b_k1) * y_k - b_k1
            s_k1[:] = (-common_term - l1_k1 / rho).clip(0, torch.inf)
            s_k2[:] = (x_k1.T + l3_k1 / rho).clip(0, torch.inf)

            # l 更新：更新拉格朗日乘子，它们对应着约束条件的违反程度
            l1_k1[:] = l1_k1 + rho * (common_term + s_k1)

            sum_x2 = torch.matmul(B, x_k1).T  # T*I
            l2_k1[:] = l2_k1 + rho * (sum_x2 - D)

            l3_k1[:] = l3_k1 + rho * (x_k1.T - s_k2)

            # square_error.append(square_error2)
            # 每次迭代后，记录当前的目标值 b_k1（带宽总和），用于后续的收敛分析和绘图
            lst_obj.append(torch.sum(b_k1).cpu() * 1000)  # convert back to CPU and scale to original units
            lst_rho.append(rho)

            # 收敛条件：当目标值的变化率小于 1e-6，并且所有约束条件（客户需求和站点容量）都得到满足时，停止迭代
            if AUTO_STOP:
                if iteration >= 2 and torch.abs((lst_obj[-1] - lst_obj[-2]) / lst_obj[-1]) <= 1e-6 and torch.eq(torch.matmul(B, x_k1), D.T).all() and torch.le(torch.floor(A1X.clip(0, torch.inf)), C.repeat(T, 1).T).all():
                    print('UB iteration', iteration + 1)
                    break
    print('rho='+str(lst_rho[-1])+' iteration='+str(max_iter))
    print('time(s): ', time.time() - start)
    print('95 value:  ', lst_obj[-1])
    plt.plot(lst_obj)
    plt.title('obj: ' + str(lst_obj[-1].cpu().numpy())+"rho="+str(rho))
    # 绘制横线：y=351.509，颜色红色（'r'），线宽2像素，线型为实线（'-'）
    # plt.axhline(y=351.509, color='r', linewidth=0.2, linestyle='-') #  8928
    plt.axhline(y=305071, color='r', linewidth=0.2, linestyle='-')  # 288 (scaled to original units)
    np.savetxt(filename, lst_obj)

    if GPU:
        plt.savefig('experiment_results/result.png')
    else:
        plt.show()

    # 保存解的详细信息用于分析
    solution_data = {
        'x_variables': x_k1.cpu().numpy().tolist(),  # 流量变量 (I*J, T)
        'b_variables': b_k1.cpu().numpy().tolist(),  # 带宽变量 (J,)
        'site_names': site_names[:J].tolist(),
        'customer_names': customer_names[:I].tolist(),
        'capacity': capacity[:J].tolist(),
        'is_link': is_link[:J, :I].tolist(),
        'I': I,
        'J': J,
        'T': T,
        'L': L,
        'objective_value': lst_obj[-1].cpu().numpy().item(),
        'iterations': len(lst_obj),
        'rho': rho
    }
    
    # 保存到JSON文件
    with open('experiment_results/eadmm_solution_detailed.json', 'w') as f:
        json.dump(solution_data, f, indent=2)
    
    print(f"解的详细信息已保存到 experiment_results/eadmm_solution_detailed.json")
    print(f"数据规模: {I}个客户, {J}个数据中心, {T}个时间点")
    print(f"目标值: {lst_obj[-1]:.2f}")

    # x_val = x_k1.cpu().numpy()
    # x_val = x_val.reshape((I, J, T))
    # x_jt = np.sum(x_val, axis=0)
    # cal_95 = 0
    # for j in range(J):
    #     cal_95 += sorted(x_jt[j, :])[-15]
    #
    # print('claculated 95:  ', cal_95)



if __name__ == '__main__':
    # print(torch.__version__)
    eadmm(RHO=0.01,iter=1000)
    # lst_RHO = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    # lst_RHO = [0.3, 0.5, 0.7]
    # for RHO in lst_RHO:
    #     eadmm(RHO)