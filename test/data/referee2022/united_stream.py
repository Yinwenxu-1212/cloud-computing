import csv

from error_code import RefereeExcpt


class UnitedStream:
    """
        客户节点在一个月内各个时刻的流量总需求
        """

    def __init__(self):
        # 初赛流量数据集，客户节点在一个月内各个时刻的流量总需求
        # {customer1:[stream1, stream2, ...],customer2:[stream1, stream2, ...]...}
        self.united_streams = dict()

    """
    解析客户流量需求信息
    输入参数：file_name, 输入文件名
    输出：    
        united_streams 每个客户端流量需求列表（初赛）
        格式为{上海: [流量1, 流量2, ...}
    """

    def parse_stream(self, file_name):
        with open(file_name, "rt") as f_csv:
            lines = csv.reader(f_csv)
            header_line = next(lines)
            for customer in header_line[1:]:
                self.united_streams[customer] = []

            time_interval = 0
            for line in lines:
                time_interval += 1
                try:
                    index = 1
                    for stream in line[1:]:
                        self.united_streams[header_line[index]].append(int(stream))
                        index += 1
                except ValueError:
                    # print("value is not int ")
                    raise RefereeExcpt(16)
        return time_interval

if __name__ == '__main__':
    stream = UnitedStream()
    time_interval = stream.parse_stream("../data2/ip1_united_streams.csv")
    print(time_interval)
    # print(stream.united_streams)
    # for key in stream.united_streams:
    #     print(key + ': ' + str(stream.united_streams[key]))
