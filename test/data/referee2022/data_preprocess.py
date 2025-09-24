import csv
import string

customer_id_map = {}
site_id_map = {}
stream_id_map = {}

def get_id_from_index(index):
    ch_list = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    total_size = len(ch_list)
    id = ""
    index -= 1  # index starts from 1
    if index < total_size:
        id += ch_list[index]
        return id
    while index > 0:
        if index < total_size:
            id += ch_list[index-1]
            return ''.join(reversed(id))
        remainder = index % total_size
        id += ch_list[int(remainder)]
        index = int(index / total_size)
    return ''.join(reversed(id))


def preporcess_qos(file_name):
    with open(file_name, mode="rt") as csv_old, open(file_name.rstrip(".csv") + "_new.csv", mode="w",
                                                      newline='') as csv_new:
        csv_reader = csv.reader(csv_old)
        csv_writer = csv.writer(csv_new)
        header_line = next(csv_reader)
        for index, customer in enumerate(header_line[1:], 1):
            customer_id_map[customer] = get_id_from_index(index)
            header_line[index] = get_id_from_index(index)
        csv_writer.writerow(header_line)

        for index, line in enumerate(csv_reader, 1):
            line[0] = site_id_map.get(line[0])
            csv_writer.writerow(line)


def preporcess_bw(file_name):
    with open(file_name, mode="rt") as csv_old, open(file_name.rstrip(".csv") + "_new.csv", mode="w",
                                                    newline='') as csv_new:
        csv_reader = csv.reader(csv_old)
        csv_writer = csv.writer(csv_new)
        line = next(csv_reader)
        line[0] = 'date'
        line[1] = 'site_name'
        line[2] = 'site_type'
        line[3] = 'bandwidth'
        new_line = [line[1], line[3]]
        csv_writer.writerow(new_line)
        for index, line in enumerate(csv_reader, 1):
            # skip center node
            if line[2] == "Center":
                continue
            site_id_map[line[1]] = get_id_from_index(index)
            line[1] = get_id_from_index(index)
            new_line = [line[1], line[3]]
            csv_writer.writerow(new_line)

def preporcess_unitedstream(file_name):
    with open(file_name, mode="rt") as csv_old, open(file_name.rstrip(".csv") + "_new.csv", mode="w",
                                                     newline='') as csv_new:
        csv_reader = csv.reader(csv_old)
        csv_writer = csv.writer(csv_new)
        header_line = next(csv_reader)
        for index, customer in enumerate(header_line[1:], 1):
            header_line[index] = customer_id_map.get(customer)
        # print(index)
        csv_writer.writerow(header_line)
        for line in csv_reader:
            csv_writer.writerow(line)


def preporcess_detailedstream(file_name):
    with open(file_name, mode="rt") as csv_old, open(file_name.rstrip(".csv") + "_new.csv", mode="w",
                                                     newline='') as csv_new:
        csv_reader = csv.reader(csv_old)
        csv_writer = csv.writer(csv_new)
        header_line = next(csv_reader)
        for index, customer in enumerate(header_line[2:], 2):
            header_line[index] = customer_id_map.get(customer)
        csv_writer.writerow(header_line)
        index = 1
        for line in csv_reader:
            if line[1] in stream_id_map:
                line[1] = stream_id_map.get(line[1])
            else:
                stream_id_map[line[1]] = get_id_from_index(index)
                line[1] = get_id_from_index(index)
                index += 1

            csv_writer.writerow(line)

def data_preprocess(qos_file, bandwidth_file, united_stream_file, detailed_stream_file):
    preporcess_bw(bandwidth_file)
    preporcess_qos(qos_file)
    preporcess_unitedstream(united_stream_file)
    preporcess_detailedstream(detailed_stream_file)


if __name__ == '__main__':
    data_preprocess("../data/QoS.csv",
                    "../data/construction_bandwidth.csv",
                    "../data/ip1_united_streams.csv",
                    "../data/ip2_detailed_streams.csv")
