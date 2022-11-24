import numpy as np
# from biosiglive.interfaces.vicon_interface import ViconClient
from biosiglive.io.save_data import add_data_to_pickle, read_data
from time import sleep, time
from custom_interface import MyInterface
from biosiglive import (LivePlot, PlotType)

if __name__ == "__main__":
    try_offline = True

    output_file_path = "trial_x.bio"
    if try_offline:
        interface = MyInterface(system_rate=100)
    else:
        # init trigno community client
        interface = ViconClient(ip="localhost", system_rate=100)

    # Add markerSet to Vicon interface
    n_markers = 15

    # Add device to Vicon interface
    interface.add_marker_set(nb_markers=n_markers,
                            data_buffer_size=100,
                         name="markers",
                         rate=100,)

    # Add plot
    marker_plot = LivePlot(name="markers", plot_type=PlotType.Scatter3D)
    marker_plot.init()

    time_to_sleep = 1 / 100

    offline_count = 0
    mark_to_plot = []
    while True:
        tic = time()
        mark_tmp, _ = interface.get_markers_data()

        marker_plot.update(mark_tmp[:, :, -1].T, size=0.1)

        loop_time = time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(real_time_to_sleep)
