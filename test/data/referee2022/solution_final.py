from bandwidth import BandWidth
from detailed_stream import DetailedStream
from error_code import RefereeExcpt
from predefs import BANDWITH_FILE
from predefs import DETAILED_STREAM_FILE
from predefs import QOS_FILE
from predefs import SOLUTION_FILE
from qos import Qos
from config import read_config
import re

class SolutionFinal:
    """
        客户节点一个月内每五分钟在各个边缘节点流量分配
        """

    def __init__(self):
        # 一个月的客户流量分配
        # 列表长度8928
        # [{customer1:{site1:[stream_id1,...], site2:[stream_id2,...],...},
        #  customer2:{site3:[stream_id3,...], site4:[stream_id4,...],...},
        #  ...},the second interval, ...]
        self.customer_edges_list = list()
        # 列表长度8928,stream用set可能会更好一些，可以快速判断流是否在里面
        # [{site1:{stream1:maxVal, stream2:maxVal,...}, site2:{stream1:maxVal, stream2:maxVal,...},...},
        # ...]
        self.center_edges_streams_list = list()
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
            if total_lines % time_intervals_num != 0:
                raise RefereeExcpt(3)
            customer_num = int(total_lines / time_intervals_num)
            if customer_num != stream_customer_num:
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
                    # customer = line
                    # customer_edge_one_charge[customer] = dict()
                    line_index += 1
                    continue

                customer_sites = line.split(':')
                customer = customer_sites[0]

                sites = customer_sites[1]
                # customer:
                if len(sites) < 5:
                    line_index += 1
                    # customer_edge_one_charge[customer] = dict()
                    continue

                # sites = sites.split('>,')
                sites = re.split('> *,', sites)

                for site in sites:
                    tmp_site_stream = site.strip(',< >').split(',')
                    tmp_site = tmp_site_stream[0]
                    tmp_stream = tmp_site_stream[1:]
                    if tmp_site not in edges_stream_dict:
                        edges_stream_dict[tmp_site] = list()

                    edges_stream_dict[tmp_site] += tmp_stream

                customer_edge_one_charge[customer] = edges_stream_dict
                line_index += 1

    def check_and_calc_cost(self, bandwidth, qos, ip_streams, time_intervals_num):
        # print(len(self.customer_edges_list))
        if len(self.customer_edges_list) != time_intervals_num:
            raise RefereeExcpt(3)
        charge_base = int(read_config('config', 'base_cost'))
        center_charge_factor = float(read_config('config', 'center_cost'))

        solution_cost_dict = dict()
        for index, one_interval in enumerate(self.customer_edges_list):
            # 汇总每5分钟每个边缘节点上的流量总和
            # {site1:stream_acc1, site2:stream_acc2, ...}
            solution_edge_streams_dict = dict()

            center_site_streams_dict = dict()
            self.center_edges_streams_list.append(center_site_streams_dict)

            # 5分钟内客户流量分配情况
            # {customer1:{stream_id1, stream_id2, stream_id3,...}, customer2:{stream_id1,...}, ...}
            solution_customer_streams_dict = dict()
            for customer, edge_nodes in one_interval.items():

                solution_customer_streams_dict[customer] = set()

                for edge_node, stream_list in edge_nodes.items():
                    if edge_node not in bandwidth.edge_node_bw.keys():
                        # print(edge_node, customer, index)
                        raise RefereeExcpt(7)

                    if edge_node not in center_site_streams_dict:
                        center_site_streams_dict[edge_node] = dict()

                    # print(edge_node, stream_id)
                    for stream_id in stream_list:
                        if stream_id not in ip_streams.detailed_streams[customer][index].keys():
                            # print(stream_id, customer, index)
                            raise RefereeExcpt(13)
                        # solution.txt去掉中心节点流量信息，改为自动分配
                        # #检查边缘节点是否已经在中心节点分配了该流
                        # if stream_id not in self.center_edges_streams_list[index][edge_node]:
                        #     raise RefereeExcpt(10)

                        # get stream value by stream_id
                        stream_val = int(ip_streams.detailed_streams[customer][index][stream_id])
                        if edge_node in solution_edge_streams_dict:
                            solution_edge_streams_dict[edge_node] += stream_val
                        else:
                            solution_edge_streams_dict[edge_node] = stream_val

                        # 记录边缘节点上每个流的最大值，该值为边缘节点在中心节点上为该流分配的流量
                        if stream_id not in self.center_edges_streams_list[index][edge_node]:
                            self.center_edges_streams_list[index][edge_node][stream_id] = stream_val
                        else:
                            self.center_edges_streams_list[index][edge_node][stream_id] = \
                                max(self.center_edges_streams_list[index][edge_node][stream_id], stream_val)

                        # 汇总一个时间周期内为客户分配的流量, 每个流只能分配到一个边缘节点
                        if stream_id in solution_customer_streams_dict[customer]:
                            raise RefereeExcpt(10)
                        solution_customer_streams_dict[customer].add(stream_id)

                    # 1.检查流量分配是否满足QOS要求
                    if edge_node not in qos.customer_edge_dict[customer]:
                        # print(edge_node, customer)
                        raise RefereeExcpt(4)

            # 2. 检查当前计费点是否为所有客户端都分配了流量
            for customer, all_streams in ip_streams.detailed_streams.items():
                customer_total_stream = 0
                for stream, value in all_streams[index].items():
                    customer_total_stream += value
                if customer not in one_interval:
                    if customer_total_stream != 0:
                        raise RefereeExcpt(5)

            # 3.检查每个客户分配的流量是否等于请求流量
            for customer, streams in solution_customer_streams_dict.items():
                for stream_id, value in ip_streams.detailed_streams[customer][index].items():
                    if value != 0 and stream_id not in streams:
                        # print(customer, stream_id, index)
                        # print(ip_streams.detailed_streams[customer][index])
                        # print(streams)
                        raise RefereeExcpt(6)

            # 先计算每个边缘节点上个时间段的残留带宽,有些边缘节点下个周期可能不会分配, 需要先保存下来
            if index > 0:
                for prev_edge_node, prev_stream_list in solution_cost_dict.items():
                    prev_stream_list[index] = prev_stream_list[index - 1] * 5 // 100
            # 4.检查每个5分钟时间间隔内每个边缘节点流量是否超标
            for edge_node, streams_acc in solution_edge_streams_dict.items():
                # 统计边缘节点95计费信息
                if edge_node not in solution_cost_dict:
                    solution_cost_dict[edge_node] = [0] * time_intervals_num

                # 在残留流量的基础上累加当期流量
                solution_cost_dict[edge_node][index] += streams_acc

                if edge_node in bandwidth.edge_node_bw:
                    # print(streams, bandwidth.edge_node_bw[edge_node])
                    if streams_acc > bandwidth.edge_node_bw[edge_node]:
                        # print(streams_acc,edge_node, bandwidth.edge_node_bw[edge_node])
                        raise RefereeExcpt(8)
                else:
                    raise RefereeExcpt(14)

        # 6.计算每个边缘节点95计费cost
        total_cost = 0.0
        reverse_charge_point = int(time_intervals_num * 0.05)  # 30天432，31天446
        for edge, stream_list in solution_cost_dict.items():
            current_edge_cost = 0
            if len(stream_list) > reverse_charge_point:
                stream_list.sort(reverse=True)  # 从大到小排序不用考虑没分配的情况
                current_edge_cost = float(stream_list[reverse_charge_point])

            # 只要使用了就分段计费
            if len(stream_list) > 0 and stream_list[0] > 0:
                if current_edge_cost <= charge_base:   # x<=A , result = A
                    current_edge_cost = charge_base
                else:  # x>A, result = pow(x-a)/B + x
                    current_edge_cost = pow(current_edge_cost - charge_base, 2)/bandwidth.edge_node_bw[edge] + current_edge_cost
            total_cost += current_edge_cost

        # 7.计算中心节点95计费cost
        center_cost_list = list()
        for index, one_interval in enumerate(self.center_edges_streams_list):
            center_streams_acc = 0
            # 取每个边缘节点每个流的最大值
            for one_site_streams in one_interval.values():
                for stream_max in one_site_streams.values():
                    center_streams_acc += stream_max

            center_cost_list.append(center_streams_acc)
            # print(center_streams_acc/1024)
        center_cost_list.sort(reverse=True)
        if len(center_cost_list) > reverse_charge_point:
            center_cost = float(center_cost_list[reverse_charge_point])
        else:
            center_cost = 0.0

        # 中心节点和边缘节点带宽成本为10:7
        # print("edge=", int(total_cost))
        # print("center=", int(center_cost * 10 / 7))
        total_cost += center_cost * center_charge_factor
        # return round(total_cost, 4)
        # 四舍五入取整
        # return Decimal(str(total_cost)).quantize(Decimal('1'), rounding = "ROUND_HALF_UP")
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

            streams = DetailedStream()
            time_intervals_num = streams.parse_stream(DETAILED_STREAM_FILE)
            # for key in stream.united_streams:
            #     print(key + ': ' + str(stream.united_streams[key]))
        except:
            raise RefereeExcpt(16)

        try:
            self.parse_solution(SOLUTION_FILE, time_intervals_num, len(streams.detailed_streams.keys()))
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
    solution = SolutionFinal()
    # solution.parse_solution('../data/solution-final.txt')
    # print(len(solution.center_edges_streams_list))
    # for interval in solution.center_edges_streams_list:
    #     print(interval)
    # for interval in solution.customer_edges_list:
    #     print(interval)
    cost = solution.calc_cost()
    print(cost)
