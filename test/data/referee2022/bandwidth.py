import csv
from error_code import RefereeExcpt

class BandWidth:
    """
        中心节点和边缘节点带宽信息
    """

    def __init__(self):
        # 数据格式
        # {site1: bw1, site2:bw2, ...}
        self.edge_node_bw = dict()  # 边缘节点带宽
        self.center_node_bw = dict()  # 中心节点带宽

    """
    解析中心节点和边缘节点带宽建设信息
    数据格式：日期, 节点名称, 节点类型, 节点直播出流基准能力GB
    输入参数：file_name, csv文件名
    输出：
        edge_node_bw 边缘节点和其对应的带宽生成的字典
        center_node_bw 中心节点和其对应的带宽生成的字典
        格式为{'CHN-AHhefei-CMCC1': 81920.0, 'CHN-AHhuainan-CMCC4': 87040.0,...}
    """

    def parse_bandwidth(self, file_name):
        with open(file_name, "rt") as f_csv:
            lines = csv.reader(f_csv)
            line = next(lines)
            for line in lines:
                try:
                   self.edge_node_bw[line[0]] = int(line[1])
                except ValueError:
                    # print("value is not float ")
                    RefereeExcpt(16)


if __name__ == '__main__':
    bw = BandWidth()
    bw.parse_bandwidth("../data/construction_bandwidth.csv")
    for key in bw.edge_node_bw:
        print(key + ': ' + str(bw.edge_node_bw[key]))
