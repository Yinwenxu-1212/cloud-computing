from bandwidth import BandWidth
from error_code import RefereeExcpt
from predefs import BANDWITH_FILE
from predefs import QOS_FILE
from predefs import SOLUTION_FILE
from predefs import UNITED_STREAM_FILE
from qos import Qos
from united_stream import UnitedStream
# from decimal import Decimal
import re
import logger_common

LOGGER = logger_common.init_log("referee.log", "./")


class SolutionPre:
    """
        客户节点一个月内每五分钟在各个边缘节点流量分配
        """

    def __init__(self):
        # 一个月的客户流量分配
        # 列表长度8928
        # [{customer1:{site1:data_assigned1, site2:data_assigned2,...},
        #  customer2:{site3:data_assigned3, site4:data_assigned4,...},
        #  ...},the second interval, ...]
        self.customer_edges_list = list()
        self.cost = 0

    def parse_solution(self, file_name, time_intervals_num, stream_customer_num):
        with open(file_name, encoding='utf8') as f_sol:
            lines = f_sol.read().splitlines()
            line_index = 0
            total_lines = 0
            for line in lines:
                # 计算行数，忽略文件末尾的空行
                if len(line.strip()) != 0:
                    total_lines += 1
            if int(total_lines % time_intervals_num) != 0:
                LOGGER.error(
                    "total_lines:" + str(total_lines) + ", time_intervals_num:" + str(
                        time_intervals_num))
                raise RefereeExcpt(3)
            customer_num = int(total_lines / time_intervals_num)
            if customer_num != stream_customer_num:
                LOGGER.error(
                    "customer number not qeual:" + "alloc:" + str(customer_num) + ", demand customer num:" + str(stream_customer_num)
                    + ", output total lines;"+ str(total_lines))
                raise RefereeExcpt(3)
            for line in lines:
                # 遇到空行就结束，如果空行在文件结尾之前表示格式错误，如果在文件结尾就忽略
                if len(line.strip()) == 0:
                    return
                # 一个计费间隔（5分钟）的客户流量分配
                if int(line_index % customer_num) == 0:
                    customer_edge_one_charge = dict()
                    self.customer_edges_list.append(customer_edge_one_charge)
                edges_stream_dict = dict()
                # 客户未分配流量，只有客户名称,有的选手写冒号，有的又没写
                if len(line) < 5 and ':' not in line:
                    customer = line.strip("' ")
                    customer_edge_one_charge[customer] = edges_stream_dict
                    line_index += 1
                    continue

                customer_sites = line.split(':')
                customer = customer_sites[0]

                sites = customer_sites[1]

                if len(sites) < 5:
                    customer_edge_one_charge[customer] = edges_stream_dict
                    line_index += 1
                    continue

                # sites = sites.split('>,')
                sites = re.split('> *,', sites)

                for site in sites:
                    tmp_site_stream = site.strip(',< >').split(',')
                    tmp_site = tmp_site_stream[0]
                    # 需不需要支持一个site对应多个流？
                    # tmp_stream = tmp_site_stream[1]
                    tmp_streams = tmp_site_stream[1:]
                    if tmp_site not in edges_stream_dict:
                        edges_stream_dict[tmp_site] = 0
                    for tmp_stream in tmp_streams:
                        try:
                            stream_val = int(tmp_stream.strip("' "))
                        except ValueError:
                            RefereeExcpt(17)
                        # 有些选手分配负的流量值
                        if (stream_val < 0):
                            RefereeExcpt(17)
                        edges_stream_dict[tmp_site] += stream_val

                customer_edge_one_charge[customer] = edges_stream_dict
                line_index += 1

    def check_and_calc_cost(self, bandwidth, qos, ip_streams, time_intervals_num):
        if len(self.customer_edges_list) != time_intervals_num:
            LOGGER.error(
                "customer_edges_list:" + str(len(self.customer_edges_list)) + ", time_intervals_num:" + str(
                    time_intervals_num))
            raise RefereeExcpt(3)

        solution_cost_dict = dict()

        for index, one_interval in enumerate(self.customer_edges_list):
            # 汇总每5分钟每个边缘节点上的流量总和
            solution_edge_streams_dict = dict()
            solution_customer_streams_dict = dict()

            for customer, edge_nodes in one_interval.items():

                solution_customer_streams_dict[customer] = 0

                for edge_node, data_assigned in edge_nodes.items():
                    if edge_node not in bandwidth.edge_node_bw.keys():
                        raise RefereeExcpt(14)

                    if edge_node in solution_edge_streams_dict:
                        solution_edge_streams_dict[edge_node] += data_assigned
                    else:
                        solution_edge_streams_dict[edge_node] = data_assigned

                    # 1.检查流量分配是否满足QOS要求
                    if edge_node not in qos.customer_edge_dict[customer]:
                        LOGGER.error("bandwidth alloc not satisfy qos:" + "edge node:" + str(edge_node) + ", customer:" + customer)
                        raise RefereeExcpt(4)

                    # 累加一个时间周期内为客户分配的流量总和
                    solution_customer_streams_dict[customer] += data_assigned

            # 2. 检查当前计费点是否为所有客户端都分配了流量
            for customer in ip_streams.united_streams.keys():
                if customer not in one_interval:
                    raise RefereeExcpt(5)

            # 3.检查每个客户分配的流量是否等于请求流量
            for customer, streams in solution_customer_streams_dict.items():
                if (ip_streams.united_streams[customer][index] != streams):
                    LOGGER.error("bandwidth alloc not equal requre, bandwidth alloc:" + str(streams) + ", requre:" + str(
                        ip_streams.united_streams[customer][index]) + ", customer:" + customer)
                    raise RefereeExcpt(6)

            # 4.检查每个5分钟时间间隔内每个边缘节点流量是否超标
            for edge_node, streams in solution_edge_streams_dict.items():
                if edge_node in bandwidth.edge_node_bw:
                    if streams > bandwidth.edge_node_bw[edge_node]:
                        LOGGER.error("site node bandwith exceed, bandwidth alloc:" + str(streams) + ", edge bw:" + str(
                            bandwidth.edge_node_bw[edge_node]) + ", edge name:" + edge_node)
                        raise RefereeExcpt(8)
                else:
                    raise RefereeExcpt(14)
                # 统计95计费信息
                if edge_node not in solution_cost_dict:
                    solution_cost_dict[edge_node] = list()
                solution_cost_dict[edge_node].append(streams)

        # 5.计算每个边缘节点95计费cost
        total_cost = 0.0
        reverse_charge_point = int(time_intervals_num * 0.05)  # 30天432，31天446
        for stream_list in solution_cost_dict.values():
            if len(stream_list) > reverse_charge_point:
                stream_list.sort(reverse=True)  # 从大到小排序不用考虑没分配的情况
                total_cost += float(stream_list[reverse_charge_point])
        # return Decimal(str(total_cost)).quantize(Decimal('1'), rounding = "ROUND_HALF_UP")
        # 四舍五入取整
        return int(total_cost + 0.5)

    def calc_cost(self):
        try:
            bw = BandWidth()
            bw.parse_bandwidth(BANDWITH_FILE)
            # for key in bw.edge_node_bw:
            #     print(key + ': ' + str(bw.edge_node_bw[key]))

            qos = Qos()
            qos.parse_qos(QOS_FILE)
            # for key in qos.customer_edge_dict:
            #     print(key + ': ' + str(qos.customer_edge_dict[key]))

            streams = UnitedStream()
            time_intervals_num = streams.parse_stream(UNITED_STREAM_FILE)
            # for key in stream.united_streams:
            #     print(key + ': ' + str(stream.united_streams[key]))
        except:
            raise RefereeExcpt(16)

        try:
            self.parse_solution(SOLUTION_FILE, time_intervals_num, len(streams.united_streams.keys()))
        except RefereeExcpt:
            raise
        except FileNotFoundError:
            raise RefereeExcpt(12)
        except PermissionError:
            raise RefereeExcpt(12)
        except Exception as err:
            raise RefereeExcpt(3)
        # print(str(self.customer_edges_list[0]))
        # for one_charge in self.customer_edges_list:
        #     for customer in one_charge:
        #         print(customer+":"+ str(one_charge[customer]))
        try:
            cost = self.check_and_calc_cost(bw, qos, streams, time_intervals_num)
        except RefereeExcpt:
            raise
        except Exception as err:
            raise RefereeExcpt(3)
        return cost


if __name__ == '__main__':
    solution = SolutionPre()
    cost = solution.calc_cost()
    print(cost)
