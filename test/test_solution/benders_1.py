# Institution：SRIBD
# Name: Yandong He
# TIME: 2024/9/22  15:33

from gurobipy import *
import random
import numpy as np



#  MIP 调用GUROBI 求解

def solve_original_MIP():
    """ Build the original MIP"""
    Original_MIP_model = Model('Benders decomposition')
    Original_MIP_model.setParam("OutputFlag", 0)

    """ create decision variables """
    y = Original_MIP_model.addVar(lb=0, ub=1000, vtype=GRB.INTEGER, name='y')
    x = {}

    for i in range(10):
        x[i] = Original_MIP_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="x_" + str(i))

    """ set objective function """
    obj = LinExpr()
    obj.addTerms(1.045, y)
    for i in range(10):
        obj.addTerms(1 + 0.01 * (i + 1), x[i])
    Original_MIP_model.setObjective(obj, GRB.MAXIMIZE)

    """ add constraints """
    lhs = LinExpr()
    lhs.addTerms(1, y)
    for i in range(10):
        lhs.addTerms(1, x[i])
    Original_MIP_model.addConstr(lhs <= 1000, name="budget")

    Original_MIP_model.optimize()

    print('Obj:', Original_MIP_model.ObjVal)
    print("Saving account", y.x)
    for i in range(10):
        if x[i].x > 0:
            print('Fund ID {}: amount: {}'.format(i + 1, x[i].x))

class MasterProblem (object):

    def __init__(self):
        self.masterModel = Model("Master Problem")
        self.masterModel.setParam("OutputFlag", 0)
        # 定义变量
        self.y = self.masterModel.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="y")
        self.z = self.masterModel.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="z")
        self.f = 1.045
        self.masterModel.setObjective(self.z, GRB.MAXIMIZE)

    def solveMP (self):
        self.masterModel.optimize()
        obj = self.masterModel.ObjVal
        print("主问题目标值：", obj)
        print("变量取值: {} = {}".format(self.y.varName, self.y.x))

        return obj, self.y.x

    def addConstraint (self,dualSP_status, B, b, alpha_combine):
        if dualSP_status == 2:
            # 添加最优cut
            lhs = LinExpr()
            lhs = self.f * self.y
            for i in range(11):
                lhs = lhs + alpha_combine[i] * (b[i] - B[i] * self.y)

            print("Current variables in master model:")
            for v in self.masterModel.getVars():
                print(f"{v.varName}: {v.x}")

            self.masterModel.addConstr(lhs >= self.z, name='benders optimality cut')
        else:
            # 添加 可行 cut
            lhs_2 = LinExpr()
            for i in range(11):
                lhs_2 = lhs_2 + alpha_combine[i] * (b[i] - B[i] * self.y)
            self.masterModel.addConstr(lhs_2 >= 0, name='benders feasibility cut')


def solveDualSP(y_0):

    # 定义变量
    Dual_SP = Model("Dual SP")
    alpha_0 = Dual_SP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha_0")
    alpha = {}
    for i in range(10):
        alpha[i] = Dual_SP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha_" + str(i))
    "create objective function "

    obj = LinExpr()
    obj.addTerms(1000 - y_0, alpha_0)
    for i in range(10):
        obj.addTerms(100, alpha[i])

    Dual_SP.setObjective(obj, GRB.MINIMIZE)

    "  add constraints 1- 10"
    for i in range(10):
        Dual_SP.addConstr(alpha_0 + alpha[i] >= 1 + 0.01 * (i + 1))

    Dual_SP.setParam("OutputFlag", 0)
    Dual_SP.setParam("InfUnbdInfo", 1)

    Dual_SP.optimize()
    obj = Dual_SP.ObjVal
    status = Dual_SP.status

    if status == 2:
        # 获得极点
        print("Dual SP Obj :", Dual_SP.ObjVal)
        alpha_0 = alpha_0.x
        for i in range(10):
            if alpha[i].x > 0:
                alpha[i] = alpha[i].x
                #print("{} = {}".format(alpha[i].varName, alpha[i].x))
    else:
        # 获得极射线
        #print("extreme ray:{} = {}".format(alpha_0.varName, alpha_0.UnbdRay))
        alpha_0 = alpha_0.UnbdRay
        for i in range(10):
            #print("extreme ray : {} = {}".format(alpha[i].varName, alpha[i].UnbdRay))
            alpha[i] = alpha[i].UnbdRay



    return status, alpha_0, alpha, obj


print("***********************************************")
print("**************   调用GUROBI求解   *****************")
print("***********************************************")
solve_original_MIP()


#  benders 求解

print("***********************************************")
print("**************   Benders求解   *****************")
print("***********************************************")

# 初始化 数值
y_0 = 1500
f = 1.045
B = []
B.append(1)
for i in range(10):
    B.append(0)
b = []
b.append(1000)
for i in range(10):
    b.append(100)

max_iteration = 100
UB = 10000
LB = 0
Gap =  UB - LB
UB_change = []
LB_change = []
EPS = 0.05  # gap
optimal_val = 0  # 模型最优值


mp = MasterProblem()
mp.solveMP()

for i in range(max_iteration):
    print("上界：", UB)
    print("下界：", LB)
    print("第 ", i, "次 benders 迭代")
    if UB - LB <= EPS:
        optimal_val = UB or LB
        print("模型的最优值：", optimal_val)
        break
    else:
        UB_change.append(UB)
        LB_change.append(LB)
        # 求解子问题
        dualSP_status, alpha_0, alpha, obj_sp = solveDualSP(y_0)
        alpha_combine = []
        alpha_combine.append(alpha_0)
        for i in range(10):
            alpha_combine.append(alpha[i])
        if dualSP_status == 2:
            # 如果子问题有最优解  则给主问题增加  最优割
            LB = max(LB, obj_sp + f * y_0)
            mp.addConstraint(dualSP_status, B, b, alpha_combine)
        else:
            mp.addConstraint(dualSP_status, B, b, alpha_combine)
        # 求解主问题

        obj, y = mp.solveMP()
        UB = obj
        y_0 = y

print("the final gap = ", UB - LB)