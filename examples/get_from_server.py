from biosiglive import (
Client,
TcpClient,
DeviceType,
)

if __name__ == '__main__':
    server_ip = "127.0.0.1"
    server_port = 50000
    tcp_client = TcpClient(server_ip, server_port, read_frequency=100)
    tcp_client.add_device(5, command_name="emg", device_type=DeviceType.Emg, name="processed EMG", rate=2000)
    tcp_client.add_marker_set(15, name="markers", rate=100)
    while True:
        data = tcp_client.get_device_data(device_name="processed EMG", nb_frame_to_get=1, down_sampling={"emg": 5})
        print(data)

