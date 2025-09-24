import csv
from error_code import RefereeExcpt
from config import read_config

class Qos:
    """
        客户节点和边缘节点时延信息
    """

    def __init__(self):
        # 边缘节点列表用set是不是更好？
        # {customer1:{site1, site2, site3,...}, customer2:{site1, site2, site3,...}, ...}
        self.customer_edge_dict = dict()  # 客户节点和满足其QOS要求的编译节点列表组成的字典

    """
    解析客户和边缘节点之间的时延信息    
    数据格式：site,上海,云南,内蒙古,北京,吉林,四川,天津,宁夏,安徽,山东,山西,广东...
    输入参数：file_name, csv文件名
    输出：
        customer_edge_dict 每个客户端及与之连通的边缘节点列表组成的字典，这样组织比较方便查询
        格式为{上海: ['CHN-AHhefei-CMCC', 'CHN-AHwuhu-AREACT', ...}
    """

    def parse_qos(self, file_name):
        try:
            qos_constraint = int(read_config('config', 'qos_constraint'))
        except Exception as err:
            RefereeExcpt(9)

        with open(file_name, "rt") as f_csv:
            lines = csv.reader(f_csv)
            header_line = next(lines)
            for customer in header_line[1:]:
                # self.customer_edge_dict[customer] = []
                self.customer_edge_dict[customer] = set()

            for line in lines:
                try:
                    index = 1
                    for latency in line[1:]:
                        if int(latency) < qos_constraint:
                            self.customer_edge_dict[header_line[index]].add(line[0])
                        index += 1
                except ValueError:
                    # print("value is not int ")
                    RefereeExcpt(16)

    """
        校验客户节点和边缘节点之间的是否满足Qos要求
        输入参数：
            customer, 客户节点
            edge_node,边缘节点
        输出：
            True： 满足要求
            False：不满足要求
        """

    def check_qos_satisfied(customer, edge_node):
        if edge_node in customer_edge_dict[customer]:
            return True
        else:
            return False


if __name__ == '__main__':
    qos = Qos()
    qos.parse_qos("../data/QoS.csv")
    for key in qos.customer_edge_dict:
        print(key + ': ', len(qos.customer_edge_dict[key]))
