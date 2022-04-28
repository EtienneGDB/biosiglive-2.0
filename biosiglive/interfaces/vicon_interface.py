import numpy as np
from .param import *

try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    pass


class ViconClient:
    """
    Class for interfacing with the Vicon system.
    """
    def __init__(self, ip: str = None, port: int = 801):
        ip = ip if ip else "127.0.0.1"
        address = f"{ip}:{port}"
        print(f"Connection to ViconDataStreamSDK at : {address} ...")
        self.vicon_client = VDS.Client()
        self.vicon_client.Connect(address)
        print("Connected to Vicon.")

        # Enable some different data types
        self.vicon_client.EnableSegmentData()
        self.vicon_client.EnableDeviceData()
        self.vicon_client.EnableMarkerData()
        self.vicon_client.EnableUnlabeledMarkerData()

        a = self.vicon_client.GetFrame()
        while a is not True:
            a = self.vicon_client.GetFrame()

        self.acquisition_rate = self.vicon_client.GetFrameRate()

        self.devices = []
        self.imu = []
        self.markers = []

    def _add_device(self, name: str, type: str = "emg", rate: int = 2000):
        device_tmp = Device(name, type, rate)
        device_tmp.info = self.vicon_client.GetDeviceOutputDetails(name)
        self.devices.append(device_tmp)

    def _add_imu(self, name: str, rate: int = 148.1, from_emg: bool = False):
        self.imu.append(Imu(name, rate, from_emg=from_emg))

    def _add_markers(self, name: str = None, rate: int = 100, unlabeled: bool = False, subject_name: str = None):
        markers_tmp = MarkerSet(name, rate, unlabeled)
        markers_tmp.subject_name = subject_name if subject_name else self.vicon_client.GetSubjectNames()[0]
        markers_tmp.markers_names = self.vicon_client.GetMarkerNames(markers_tmp.subject_name) if not name else name
        self.markers.append(markers_tmp)

    @staticmethod
    def get_force_plate_data(vicon_client):
        forceVectorData = []
        forceplates = vicon_client.GetForcePlates()
        for plate in forceplates:
            forceVectorData = vicon_client.GetForceVector(plate)
            momentVectorData = vicon_client.GetMomentVector(plate)
            copData = vicon_client.GetCentreOfPressure(plate)
            globalForceVectorData = vicon_client.GetGlobalForceVector(plate)
            globalMomentVectorData = vicon_client.GetGlobalMomentVector(plate)
            globalCopData = vicon_client.GetGlobalCentreOfPressure(plate)

            try:
                analogData = vicon_client.GetAnalogChannelVoltage(plate)
            except VDS.DataStreamException as e:
                print("Failed getting analog channel voltages")
        return forceVectorData

    def get_device_data(self, device_name: str = None, channel_names: str = None):
        devices = []
        all_device_data = []
        if not isinstance(device_name, list):
            device_name = [device_name]

        if device_name:
            for d, device in enumerate(self.devices):
                if device.name == device_name[d]:
                    devices.append(device)
        else:
            devices = self.devices

        for device in devices:
            device_data = np.zeros((device.infos[0], device.sample))
            count = 0
            for output_name, chanel_name, unit in device.infos:
                data_tmp, _ = self.vicon_client.GetDeviceOutputValues(
                    device.name, output_name, chanel_name
                )
                if channel_names:
                    if chanel_name in channel_names:
                        device_data[count, :] = data_tmp
                else:
                    device_data[count, :] = data_tmp
                count += 1
            all_device_data.append(device_data)
        return all_device_data

    def get_markers_data(self, marker_names: list = None, subject_name: str = None):
        markers_set = []
        occluded = []
        all_markers_data = []
        all_occluded_data = []
        if subject_name:
            for s, marker_set in enumerate(self.markers):
                if marker_set.subject_name == subject_name[s]:
                    markers_set.append(marker_set)
        else:
            markers_set = self.markers

        for markers in markers_set:
            markers_data = np.zeros((3, len(markers.markers_names), markers.sample))
            count = 0
            for m, marker_name in enumerate(markers.markers_names):
                markers_data_tmp, occluded_tmp = self.vicon_client.GetMarkerGlobalTranslation(
                    markers.subject_name, marker_name
                )
                if marker_names:
                    if marker_name in marker_names:
                        markers_data[:, count, :] = markers_data_tmp
                        occluded.append(occluded_tmp)
                else:
                    markers_data[:, count, :] = markers_data_tmp
                    occluded.append(occluded_tmp)
                count += 1
            all_markers_data.append(markers_data)
            all_occluded_data.append(occluded)
        return all_markers_data, all_occluded_data

    # def get_imu(self, imu_names=None):  # , init=False, output_names=None, imu_names=None):
    #     # output_names = [] if output_names is None else output_names
    #     names = [] if imu_names is None else imu_names
    #     if self.devices == "vicon":
    #         imu = np.zeros((144, self.imu_sample))
    #         # if init is True:
    #         #     count = 0
    #         #     for output_name, imu_name, unit in self.imu_device_info:
    #         #         imu_tmp, occluded = self.vicon_client.GetDeviceOutputValues(
    #         #             self.imu_device_name, output_name, imu_name
    #         #         )
    #         #         imu[count, :] = imu_tmp[-self.imu_sample:]
    #         #         if np.mean(imu[count, :, -self.imu_sample:]) != 0:
    #         #             output_names.append(output_name)
    #         #             imu_names.append(imu_name)
    #         #         count += 1
    #         # else:
    #         count = 0
    #         for output_name, imu_name, unit in self.imu_device_info:
    #             imu_tmp, occluded = self.vicon_client.GetDeviceOutputValues(self.imu_device_name, output_name, imu_name)
    #             if imu_names is None:
    #                 names.append(imu_name)
    #             imu[count, :] = imu_tmp[-self.imu_sample :]
    #             count += 1
    #
    #         imu = imu[: self.nb_electrodes * 9, :]
    #         imu = imu.reshape(self.nb_electrodes, 9, -1)
    #     else:
    #         imu = self.dev_imu.read()
    #         imu = imu.reshape(self.nb_electrodes, 9, -1)
    #
    #     return imu, names

