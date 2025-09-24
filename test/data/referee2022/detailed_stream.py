import csv

from error_code import RefereeExcpt


class DetailedStream:
    """
    客户节点在一个月内各个时刻的流量总需求
    """

    def __init__(self):
        # 复赛/决赛流量数据集，各个省份细分流量需求
        # {customer1:[{str1:value, str2:value}...}, {str1:value, str2:value}...}],
        #  customer2:[{str1:value, str2:value}...}, {str1:value, str2:value}...}]...}
        self.detailed_streams = dict()
        # [{stream1:minval, stream2:minval}, {stream1:minval, strea2:min_val,...},...]
        # self.streams_minval_list = list()

    def parse_stream(self, file_name):
        with open(file_name, "rt") as f_csv:
            lines = csv.reader(f_csv)
            header_line = next(lines)

            for customer in header_line[2:]:
                self.detailed_streams[customer] = []

            mtime = None
            time_interval = -1
            for line in lines:
                try:
                    stream_name = line[1]
                    if mtime == line[0]:
                        index = 2
                        # tmp_minval_list = list()
                        for stream in line[2:]:
                            stream_map_list = self.detailed_streams[header_line[index]]
                            stream_map_list[time_interval][line[1]] = int(stream)
                            index += 1

                            # if int(stream) > 0:
                            #     tmp_minval_list.append(int(stream))
                        # 计算每个流在每个interval客户的最小值
                        # if len(tmp_minval_list) > 0:
                        #     self.streams_minval_list[time_interval][stream_name] = min(tmp_minval_list)
                    else:
                        # 每个interval，每个customer重新分配一个新的dict
                        time_interval += 1
                        for customer in header_line[2:]:
                            self.detailed_streams[customer].append(dict())

                        # self.streams_minval_list.append(dict())

                        mtime = line[0]
                        index = 2
                        # tmp_minval_list = list()
                        for stream in line[2:]:
                            stream_map_list = self.detailed_streams[header_line[index]]
                            stream_map_list[time_interval][stream_name] = int(stream)
                            index += 1

                            # if int(stream) > 0:
                            #     tmp_minval_list.append(int(stream))

                        # 计算每个流在每个interval客户的最小值
                        # if len(tmp_minval_list) > 0:
                        #     self.streams_minval_list[time_interval][stream_name] = min(tmp_minval_list)

                except ValueError:
                    # print("value is not int ")
                    raise RefereeExcpt(16)
        return time_interval + 1

if __name__ == '__main__':
    stream = DetailedStream()
    time_interval = stream.parse_stream("../data/ip2_detailed_streams.csv")
    print(time_interval)
    # for interval in stream.streams_minval_list:
    #     print(interval)
    # for key,stream_map_list in stream.detailed_streams.items():
    #     for stream in stream_map_list:
    #         print(stream)
