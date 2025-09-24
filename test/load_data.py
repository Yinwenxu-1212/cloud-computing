import pandas as pd
import numpy as np


def load_data():
    # 95* 31
    demand_file = './data/demand.csv'
    qos_file = './data/qos.csv'
    capacity_file = './data/site_bandwidth.csv'

    # load demand
    demand_df = pd.read_csv(demand_file)
    # time_df代表时间列，demand_df代表需求数据
    time_df = demand_df.iloc[:, 0]  # 时间列
    demand_df = demand_df.iloc[:, 1:]  # 需求数据
    demand_df = demand_df.sort_index(axis=1)
    # 将客户需求数据转换为numpy数组
    demand = np.array(demand_df, dtype=np.float64)
    # 将时间数据转换为numpy数组，并且格式化
    time_list = np.array(time_df)
    for i in range(time_list.size):
        time_list[i] = time_list[i].replace('T', '') + ':00'
    # 获取客户名称
    customer_names = demand_df.columns.values

    # load qos
    qos_df = pd.read_csv(qos_file)
    # 按照site_name排序
    qos_df = qos_df.sort_values(by='site_name', ascending=True)
    # 获取site_name列
    site_names = np.array(qos_df.iloc[:, 0])
    # 去掉site_name列
    qos_df = qos_df.iloc[:, 1:]
    # 按照客户名称排序
    qos_df = qos_df.sort_index(axis=1)
    # 检查客户名称是否一致
    assert (customer_names == qos_df.columns.values).all()
    # 将qos数据转换为numpy数组
    qos = np.array(qos_df, dtype=np.float64)
    # 检查qos是否小于400，is_link是一个布尔数组，维度与qos相同，用来表示是否是有效链接
    is_link = qos < 400

    # load capacity
    capacity_df = pd.read_csv(capacity_file)
    # 按照site_name排序
    capacity_df = capacity_df.sort_values(by='site_name', ascending=True)
    # 检查site_name是否一致
    assert (site_names == np.array(capacity_df.iloc[:, 0])).all()
    # 将capacity数据转换为numpy数组
    capacity = np.array(capacity_df.iloc[:, 1], dtype=np.float64)

    # 构建返回字典 
    result = {}
    result['customer_names'] = customer_names
    result['site_names'] = site_names
    result['demand'] = demand
    result['is_link'] = is_link
    result['capacity'] = capacity
    result['time_list'] = time_list

    return result


if __name__ == '__main__':
    load_data()
    # print('test')
    # print(pd.__version__)
    # print(np.__version__)
