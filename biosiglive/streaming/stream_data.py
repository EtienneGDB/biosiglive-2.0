from typing import Union
from time import time, sleep, strftime
import datetime
import numpy as np
import multiprocessing as mp
from biosiglive.streaming.server import Server
from biosiglive.io import save_data
from biosiglive.processing.msk_functions import MskFunctions
from ..interfaces.generic_interface import GenericInterface
from ..interfaces.param import Device, MarkerSet
from ..enums import InterfaceType, DeviceType, MarkerType, InverseKinematicsMethods, RealTimeProcessingMethod
from ..gui.plot import LivePlot
from .utils import dic_merger


class StreamData:
    def __init__(self, stream_rate: int = 100):
        """
        Initialize the StreamData class.
        Parameters
        ----------
        stream_rate: int
            The stream rate of the data.
        """
        self.process = mp.Process
        self.queue = mp.Queue()
        self.event = mp.Event()
        self.devices = []
        self.marker_sets = []
        self.plots = []
        self.stream_rate = stream_rate
        self.interfaces_type = []
        self.processes = []
        self.interfaces = []

        # Multiprocessing stuff
        self.device_queue = []
        self.kin_queue = []
        self.plots_queue = []
        self.server_queue_in = []
        self.server_queue_out = []
        self.device_event = []
        self.is_device_data = self.event
        self.is_kin_data = self.event
        self.interface_event = []
        self.kin_event = []
        self.custom_processes = []
        self.custom_processes_kwargs = []
        self.custom_processes_names = []
        self.custom_queue_in = []
        self.custom_queue_out = []
        self.custom_event = []
        self.save_data = None
        self.save_path = None
        self.save_frequency = None
        self.plots_multiprocess = False
        self.device_buffer_size = []
        self.marker_set_buffer_size = []
        self.raw_plot = None
        self.data_to_plot = None

        # Server stuff
        self.start_server = None
        self.server_ip = None
        self.ports = []
        self.client_type = None
        self.count_server = 0
        self.server_queue = []
        if isinstance(self.ports, int):
            self.ports = [self.ports]
        for p in range(len(self.ports)):
            self.server_queue.append(self.queue)

        self.device_decimals = 6
        self.kin_decimals = 4

    def _add_device(self, device: Device):
        """
        Add a device to the stream.
        Parameters
        ----------
        device: Device
            Device to add.
        """
        self.devices.append(device)
        self.device_queue.append(None)
        self.device_event.append(None)

    def add_interface(self, interface: callable):
        """
        Add an interface to the stream.
        Parameters
        ----------
        interface: GenericInterface
            Interface to add. Interface should inherit from the generic interface.
        """
        if self.multiprocess_started:
            raise Exception("Cannot add interface after the stream has started.")
        self.interfaces.append(interface)
        self.interfaces_type.append(interface.interface_type)
        self.interface_event.append(self.event)
        for device in interface.devices:
            self._add_device(device)
        for marker in interface.marker_sets:
            self._add_marker_set(marker)
        if len(self.interfaces) > 1:
            raise ValueError("Only one interface can be added for now.")

    def add_server(self, server_ip: str = "127.0.0.1", ports: Union[int, list] = 50000, client_type: str = "TCP",
                   device_buffer_size: Union[int, list] = None, marker_set_buffer_size: [int, list] = None):
        """
        Add a server to the stream.
        Parameters
        ----------
        server_ip: str
            The ip address of the server.
        ports: int or list
            The port(s) of the server.
        client_type: str
            The type of client to use. Can be TCP.
        device_buffer_size: int or list
            The size of the buffer for the devices.
        marker_set_buffer_size: int or list
            The size of the buffer for the marker sets.
        """
        if self.multiprocess_started:
            raise Exception("Cannot add interface after the stream has started.")
        self.server_ip = server_ip
        self.ports = ports
        self.client_type = client_type

        if not device_buffer_size:
            device_buffer_size = [None] * len(self.devices)
        if isinstance(device_buffer_size, list):
            if len(device_buffer_size) != len(self.devices):
                raise ValueError("The device buffer size list should have the same length as the number of devices.")
            self.device_buffer_size = device_buffer_size
        elif isinstance(device_buffer_size, int):
            self.device_buffer_size = [device_buffer_size] * len(self.devices)

        if not marker_set_buffer_size:
            marker_set_buffer_size = [None] * len(self.marker_sets)
        if isinstance(marker_set_buffer_size, list):
            if len(marker_set_buffer_size) != len(self.marker_sets):
                raise ValueError("The marker set buffer size list should have the same length as the number of marker sets.")
            self.marker_set_buffer_size = marker_set_buffer_size
        elif isinstance(marker_set_buffer_size, int):
            self.marker_set_buffer_size = [marker_set_buffer_size] * len(self.marker_sets)

    def start(self, save_streamed_data: bool = False, save_path: str = None, save_frequency: int = None):
        """
        Start the stream.
        Parameters
        ----------
        save_streamed_data: bool
            If True, the streamed data will be saved.
        save_path: str
            The path to save the streamed data.
        save_frequency:
            The frequency at which the data will be saved.
        """
        self.save_data = save_streamed_data
        self.save_path = save_path if save_path else f"streamed_data_{strftime('%Y%m%d_%H%M%S')}.bio"
        self.save_frequency = save_frequency if save_frequency else self.stream_rate
        self._init_multiprocessing()

    def _add_marker_set(self, marker: MarkerSet):
        """
        Add a marker set to the stream.
        Parameters
        ----------
        marker: MarkerSet
            Marker set to add from given interface.
        """
        self.marker_sets.append(marker)
        self.kin_queue.append(None)
        self.kin_event.append(None)

    def device_processing(self, device: Device, device_idx: int, **kwargs):
        """
        Process the data from the device
        Parameters
        ----------
        device: Device
            The device to process
        device_idx: int
            The index of the device in the list of devices
        kwargs: dict
            The kwargs to pass to the process method
        Returns
        -------

        """
        if not self.device_buffer_size[device_idx]:
            self.device_buffer_size[device_idx] = device.sample
        if device.process_method is None:
            raise ValueError("No processing method defined for this device.")
        while True:
            self.is_device_data.wait()
            processed_data = device.process(**kwargs)
            self.device_queue[device_idx].put({"processed_data": processed_data[: -self.device_buffer_size[device_idx]:]})
            self.device_event[device_idx].set()

    def recons_kin(self, marker_set: MarkerSet, marker_set_idx: int, **kwargs):
        """
        Reconstruct kinematics from markers.
        Parameters
        ----------
        marker_set: MarkerSet
            The marker set to reconstruct kinematics from.
        marker_set_idx: int
            Index of the marker set in the list of markers.
        Returns
        -------

        """
        if not self.marker_set_buffer_size[marker_set_idx]:
            self.marker_set_buffer_size[marker_set_idx] = marker_set.sample
        while True:
            self.is_kin_data.wait()
            states = marker_set.kin_method(**kwargs)
            self.kin_queue[marker_set_idx].put({"kinematics": states[:, -self.device_buffer_size[marker_set_idx]:]})
            self.kin_event[marker_set_idx].set()

    def open_server(self):
        """
        Open the server to send data from the devices.
        """
        server = Server(self.server_ip, self.ports[self.count_server], server_type=self.client_type)
        server.start()
        while True:
            connection, message = server.client_listening()
            data_queue = []
            while len(data_queue) == 0:
                # use Try statement as the queue can be empty and is_empty function is not reliable.
                try:
                    data_queue = self.server_queue[self.count_server].get_nowait()
                    is_working = True
                except mp.Queue().empty:
                    is_working = False
                if is_working:  # use this method to avoid blocking the server with Windows os.
                    server.send_data(data_queue, connection, message)

    def _init_multiprocessing(self):
        """
        Initialize the multiprocessing.
        """
        processes = []
        for d, device in enumerate(self.devices):
            if device.process_method is not None:
                self.processes.append(
                    self.process(
                        name=f"process_{device.name}",
                        target=StreamData.device_processing,
                        args=(
                            self,
                            device,
                            d,
                        ),
                    )
                )
        if self.start_server:
            for i in range(len(self.ports)):
                processes.append(self.process(name="listen" + f"_{i}", target=StreamData.open_server, args=(self,)))
                self.count_server += 1
        for interface in self.interfaces:
            processes.append(self.process(name="reader", target=StreamData.save_streamed_data, args=(self, interface)))

        for p, plot in enumerate(self.plots):
            for device in self.devices:
                for marker_set in self.marker_sets:
                    if self.data_to_plot[p] not in device.name or self.data_to_plot[p] not in marker_set.name:
                        raise ValueError(f"The name of the data to plot {self.data_to_plot[p]} is not correct.")
            if self.plots_multiprocess:
                processes.append(self.process(name="plot", target=StreamData.plot_update, args=(self, plot)))
            else:
                processes.append(self.process(name="plot", target=StreamData.plot_update, args=(self, self.plots)))
                break

        for m, marker in enumerate(self.marker_sets):
            if marker.kin_method:
                processes.append(
                    self.process(
                        name=f"process_{marker.name}",
                        target=StreamData.recons_kin,
                        args=(
                            self,
                            marker,
                            m,
                        ),
                    )
                )

        for i, funct in enumerate(self.custom_processes):
            processes.append(
                self.process(
                    name=self.custom_processes_names[i],
                    target=funct,
                    args=(self,),
                    kwargs=self.custom_processes_kwargs[i],
                )
            )
        for p in processes:
            p.start()
        self.multiprocess_started = True
        for p in processes:
            p.join()

    def _check_nb_processes(self):
        """
        compute the number of process.
        """
        nb_processes = 0
        for device in self.devices:
            if device.process_method is not None:
                nb_processes += 1
        if self.start_server:
            nb_processes += len(self.ports)
        nb_processes += len(self.plots)
        nb_processes += len(self.interfaces)
        for marker in self.marker_sets:
            if marker.kin_method:
                nb_processes += 1
        nb_processes += len(self.custom_processes)
        return nb_processes

    def add_plot(self, plot: Union[LivePlot, list], data_to_plot:Union[str, list], raw: Union[bool, list]=None, multiprocess=False):
        """
        Add a plot to the live data.
        Parameters
        ----------
        plot: Union[LivePlot, list]
            Plot to add.
        multiprocess: bool
            If True, if several plot each plot will be on a separate process. If False, each plot will be on the same one.
        """
        if isinstance(data_to_plot, str):
            data_to_plot = [data_to_plot]
        if isinstance(raw, bool):
            raw = [raw]
        if len(data_to_plot) != len(raw):
            raise ValueError("The length of the data to plot and the raw list must be the same.")
        if not raw:
            raw = [True] * len(data_to_plot)
        self.raw_plot = raw
        self.data_to_plot = data_to_plot
        if self.multiprocess_started:
            raise Exception("Cannot add plot after the stream has started.")
        self.plots_multiprocess = multiprocess
        if not isinstance(plot, list):
            plot = [plot]
        for plt in plot:
            if plt.rate:
                if plt.rate > self.stream_rate:
                    raise ValueError("Plot rate cannot be higher than stream rate.")
            self.plots.append(plt)

    def plot_update(self, plots: Union[LivePlot, list]):
        """
        Update the plots.

        Parameters
        ----------
        plots: Union[LivePlot, list]
            Plot to update.
        """
        if not isinstance(plots, list):
            plots = [plots]
        plot_idx = 0
        data_to_plot = None
        while True:
            for p, plot in enumerate(plots):
                for plt in self.plots:
                    if plot == plt:
                        plot_idx = p
                for device in self.devices:
                    for marker_set in self.marker_sets:
                        if self.data_to_plot[plot_idx] in device.name:
                            data_to_plot = device.processed_data if not self.raw_plot[plot_idx] else device.raw_data
                        if self.data_to_plot[plot_idx] in marker_set.name:
                            data_to_plot = marker_set.kin_data[0] if not self.raw_plot[plot_idx] else marker_set.raw_data
                plot.update(data_to_plot)

    def save_streamed_data(self, interface: GenericInterface):
        """
        Stream, process and save the data.
        Parameters
        ----------
        interface: callable
            Interface to use to get the data.

        """
        initial_time = 0
        iteration = 0
        dic_to_save = {}
        save_count = 0
        self.save_frequency = self.save_frequency if self.save_frequency else self.stream_rate
        while True:
            data_dic = {}
            proc_device_data = []
            raw_device_data = []
            raw_markers_data = []
            all_device_data = []
            all_markers_tmp = []
            kin_data = []
            tic = time()
            if iteration == 0:
                initial_time = time() - tic
            interface_latency = interface.get_latency()
            is_frame = interface.get_frame()
            absolute_time_frame = datetime.datetime.now()
            absolute_time_frame_dic = {
                "day": absolute_time_frame.day,
                "hour": absolute_time_frame.hour,
                "hour_s": absolute_time_frame.hour * 3600,
                "minute": absolute_time_frame.minute,
                "minute_s": absolute_time_frame.minute * 60,
                "second": absolute_time_frame.second,
                "millisecond": int(absolute_time_frame.microsecond / 1000),
                "millisecond_s": int(absolute_time_frame.microsecond / 1000) * 0.001,
            }
            self.is_kin_data.clear()
            self.is_device_data.clear()
            if is_frame:
                if iteration == 0:
                    print("Data start streaming")
                    iteration = 1
                if len(interface.devices) != 0:
                    all_device_data = interface.get_device_data(device_name="all", get_frame=False)
                    self.is_device_data.set()
                if len(interface.marker_sets) != 0:
                    all_markers_tmp, _ = interface.get_marker_set_data(get_frame=False)
                    self.is_kin_data.set()
                time_to_get_data = time() - tic
                tic_process = time()
                if len(interface.devices) != 0:
                    for i in range(len(interface.devices)):
                        if self.devices[i].process_method is not None:
                            self.device_event[i].wait()
                            device_data = self.device_queue[i].get_nowait()
                            self.device_event[i].clear()
                            proc_device_data.append(np.around(device_data["proc_device_data"], decimals=self.device_decimals))
                        if not self.device_buffer_size[i]:
                            buffer_size = self.devices[i].sample
                        else:
                            buffer_size = self.device_buffer_size[i]
                        raw_device_data.append(np.around(all_device_data[i][..., -buffer_size:], decimals=self.device_decimals))
                    data_dic["proc_device_data"] = proc_device_data
                    data_dic["raw_device_data"] = raw_device_data

                if len(interface.marker_sets) != 0:
                    for i in range(len(interface.marker_sets)):
                        if self.marker_sets[i].kin_method is not None:
                            self.kin_event[i].wait()
                            kin_data = self.kin_queue[i].get_nowait()
                            self.kin_event[i].clear()
                            kin_data.append(np.around(kin_data["kinematics_data"], decimals=self.kin_decimals))
                        raw_markers_data.append(np.around(all_markers_tmp[i], decimals=self.kin_decimals))
                    data_dic["kinematics_data"] = kin_data
                    data_dic["marker_set_data"] = raw_markers_data
                process_time = time() - tic_process  # time to process all data + time to get data

                for i in range(len(self.ports)):
                    try:
                        self.server_queue[i].get_nowait()
                    except mp.Queue().empty:
                        pass
                    self.server_queue[i].put_nowait(data_dic)

                data_dic["absolute_time_frame"] = absolute_time_frame_dic
                data_dic["interface_latency"] = interface_latency
                data_dic["process_time"] = process_time
                data_dic["initial_time"] = initial_time
                data_dic["time_to_get_data"] = time_to_get_data

                # Save data
                if self.save_data is True:
                    dic_to_save = dic_merger(data_dic, dic_to_save)
                    if save_count == int(self.stream_rate / self.save_frequency):
                        save_data.add_data_to_pickle(data_dic, self.save_path)
                        dic_to_save = {}
                        save_count = 0
                    save_count += 1

                if tic - time() < 1 / self.stream_rate:
                    sleep(1 / self.stream_rate - (tic - time()))
                else:
                    print(f"WARNING: Stream rate ({self.stream_rate}) is too high for the computer."
                          f"The actual stream rate is {1 / (tic - time())}")
