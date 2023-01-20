import numpy as np
from custom_interface import MyInterface
from biosiglive.gui.plot import LivePlot
from biosiglive import (
    # LivePlot,
    save,
    load,
    ViconClient,
    PytrignoClient,
    RealTimeProcessingMethod,
    RealTimeProcessing,
    PlotType,
)
from pyomeca import Analogs
from time import sleep, time
import datetime
from scipy import stats
import matplotlib.pyplot as plt
import os

# try:
#     import biorbd
# except ImportError:
#     pass


# def get_custom_function(device_interface):
#     custom_processing = RealTimeProcessing(
#         data_rate=device_interface.get_device(name="emg").rate, processing_window=1000
#     )
#     custom_processing.bpf_lcut = 10
#     custom_processing.bpf_hcut = 425
#     custom_processing.lpf_lcut = 5
#     custom_processing.lp_butter_order = 4
#     custom_processing.bp_butter_order = 2
#     custom_processing.moving_average_windows = 200
#     return custom_processing.process_emg

def envelop(device_data):
    signal = np.array(device_data)
    pyo_signal = Analogs(signal)
    pyo_signal_processed = (
        pyo_signal.meca.abs()
        .meca.low_pass(order=4, cutoff=9, freq=2000)
    )
    return np.array(pyo_signal_processed)

def threshold_activation_detection(device_data):
    mean_resting = np.array(np.mean(envelop(device_data), axis=1))
    mean_resting = mean_resting.reshape(n_electrodes, 1)
    std_resting = np.array(np.std(envelop(device_data), axis=1))
    std_resting = std_resting.reshape(n_electrodes, 1)
    tresh_muscle = mean_resting + 3*std_resting
    tresh = np.array(np.mean(tresh_muscle))
    return tresh

def median_freq(fft_signal):
    med_freq = []
    for iM in range(fft_signal.shape[0]):
        ITotal = 0
        for i in range(fft_signal.shape[1]):
            S = fft_signal.data[iM, i]
            ITotal = ITotal + S

        IT = ITotal / 2
        Iinc = 0
        for i in range(fft_signal.shape[1]):
            Ic = fft_signal.data[iM, i]
            Iinc = Iinc + Ic
            if Iinc > IT:
                med_freq.append(fft_signal.freq.data[i])
                break
    return med_freq

def process_data_fFreqMed(device_data, interval=False, tps=datetime):
    if interval:
        if tps.second == tps.second:
            print("Peut être utilise pour mettre un delai de 1 min entre chaque calcul d'indicateur")

    signal = np.array(device_data)
    pyo_signal = Analogs(signal)
    # low_cut = 10
    # high_cut = 450
    # EMGBP = pyo_signal.meca.band_pass(order=2, cutoff=[low_cut, high_cut], freq=2000)
    # EMGBP_Centered = EMGBP.meca.center()
    pyo_signal_fft = pyo_signal.meca.fft(freq=2000, only_positive=True)
    Med_Freq = np.array(median_freq(pyo_signal_fft))
    Med_Freq = Med_Freq.reshape(n_electrodes, 1)
    return Med_Freq

if __name__ == "__main__":
    try_offline = True

    output_file_path = "trial_x.bio"
    if try_offline:
        interface = MyInterface(system_rate=100, data_path="abd.bio")
        # Get prerecorded data from pickle file for a shoulder abduction
        # offline_emg = load("abd.bio")["emg"]
    else:
        # init trigno community client
        # interface = ViconClient(ip="localhost", system_rate=100)
        interface = PytrignoClient(system_rate=100, ip="127.0.0.1")


    # Add markerSet to Vicon interface
    n_electrodes = 2
    Median_Frequency = np.zeros((n_electrodes, 50))
    Baseline_Median_Frequency = np.zeros((n_electrodes, 1))
    Evolution_Median_Frequency = np.zeros((n_electrodes, 1))
    nSample_Baseline = 267 # (gives about 20 sec of data)
    Ev_MF = False
    nSample_Ev_MF = 50
    Rest_Act = True
    # t_value = [[], [], [], []]
    # p_value = [[], [], [], []]

    output_file_path = "trial_x"

    muscle_names = [
        "Left biceps",
        "Right biceps",
        # "Deltoid medial",
        # "Deltoid posterior",
    ]

    # Add device to Vicon interface
    interface.add_device(
        nb_channels=n_electrodes,
        device_type="emg",
        name="emg",
        rate=2000,
        device_data_file_key="emg",
        # processing_method=None,
        data_buffer_size=2000,
        processing_method=RealTimeProcessingMethod.ProcessEmg,
        moving_average=False,
        absolute_value=False,
        bpf_lcut=10,
        bpf_hcut=450,
    )

    # Add plot
    emg_raw_plot = LivePlot(
        name="emg_raw", rate=100, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    emg_raw_plot.init(plot_windows=10000, colors=(255, 0, 0), y_labels="EMG (mV)")

    emg_filtered_plot = LivePlot(
        name="emg_filtered", rate=100, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    emg_filtered_plot.init(plot_windows=10000, y_labels="emg_filtered")

    emg_envelop = LivePlot(
        name="emg_envelop", rate=100, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    emg_envelop.init(plot_windows=10000, colors=(255, 0, 0), y_labels="emg_envelop")

    emg_medFreq = LivePlot(name="MedFreq", rate=100, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    emg_medFreq.init(plot_windows=500, y_labels="MedFreq")

    time_to_sleep = 1 / 100
    count = 0
    # emg_filt=None
    while True:
        # if count == 500:
        #     os.system("pause")
        tic = time()
        mtn = datetime.datetime.now()
        raw_emg = interface.get_device_data(device_name="emg")
        filtered_emg = interface.devices[0].process()
        emg_raw_plot.update(raw_emg)
        # print(interface.devices[0].raw_data.shape)
        emg_filtered_plot.update(filtered_emg[:, -20:])

        if filtered_emg[0][0] != 0:
            # Define Resting Baseline threshold for activation detection
            if Rest_Act:
                start = time()
                Resting_Activation_Threshold = RealTimeProcessing.custom_processing(
                    self=RealTimeProcessing, funct=threshold_activation_detection, data_tmp=filtered_emg
                )
                Rest_Act = False

            Activation_level = RealTimeProcessing.custom_processing(
                self=RealTimeProcessing, funct=envelop, data_tmp=filtered_emg
            )
            emg_envelop.update(Activation_level[:, -20:])

            # Median Frequency real time calculation
            MF = RealTimeProcessing.custom_processing(
                self=RealTimeProcessing, funct=process_data_fFreqMed, data_tmp=filtered_emg, interval=False, tps=mtn
            )
            Median_Frequency = np.append(Median_Frequency, MF, axis=1)
            Median_Frequency = np.delete(Median_Frequency, 0, axis=1)
            Mov_Av_MedFreq = np.mean(Median_Frequency, axis=1)
            Mov_Av_MedFreq = Mov_Av_MedFreq[:, np.newaxis]

            # for iM in range(n_electrodes):
            # if Activation_level.any() > Resting_Activation_Threshold:
            # else:
            #     Median_Frequency[:] = 0

            # plt.figure()
            # for iM in range(n_electrodes):
            #     plt.subplot(2, 1, iM+1)
            #     plt.plot(Activation_level[iM, :])
            #     plt.plot([0, 200], [Resting_Activation_Threshold, Resting_Activation_Threshold])
            # plt.show()

            if Baseline_Median_Frequency.shape[1] < nSample_Baseline:
                Baseline_Median_Frequency = np.append(Baseline_Median_Frequency, Mov_Av_MedFreq, axis=1)

            if Ev_MF:
                Evolution_Median_Frequency = np.append(Evolution_Median_Frequency, Median_Frequency, axis=1)
                Evolution_Median_Frequency = np.delete(Evolution_Median_Frequency, 0, axis=1)
                Mean_Evolution_MF = Evolution_Median_Frequency[:, 0:].mean(axis=1)
                Mean_Evolution_MF = Mean_Evolution_MF.reshape(n_electrodes, 1)
                # Var_Evolution_MF = Evolution_Median_Frequency.var(axis=1)
                # Var_Evolution_MF = Var_Evolution_MF.reshape(4, 1)
                for iM in range(n_electrodes):
                    paired_t_test = stats.ttest_1samp(Baseline_Median_Frequency[iM],
                                                    np.mean(Evolution_Median_Frequency[iM]))
                    # t_value[iM].append(paired_t_test.statistic)
                    # p_value[iM].append(paired_t_test.pvalue)
                    if (paired_t_test.pvalue < 0.05) & (Mean_Baseline_MF[iM] - Mean_Evolution_MF[iM] < 0):
                        print(muscle_names[iM], "showed manifestation of muscle fatigue")
                        Baseline_Median_Frequency = Evolution_Median_Frequency[:]

            if Baseline_Median_Frequency.shape[1] == nSample_Baseline:
                Baseline_Median_Frequency = np.delete(Baseline_Median_Frequency, 0, axis=1)
                Mean_Baseline_MF = Baseline_Median_Frequency[:, 0:].mean(axis=1)
                Mean_Baseline_MF = Mean_Baseline_MF.reshape(n_electrodes, 1)
                # Var_Baseline_MF = Baseline_Median_Frequency.var(axis=1)
                # Var_Baseline_MF = Var_Baseline_MF.reshape(4, 1)
                nSample_Baseline = 0
                Ev_MF = True
                Evolution_Median_Frequency = Baseline_Median_Frequency[:, -nSample_Ev_MF:]



            emg_medFreq.update(Mov_Av_MedFreq[:, -1:])






        # if emg_filt is not None:
        #     emg_filt = np.append(emg_filt, filtered_emg[:, -20:], axis=1)
        # else:
        #     emg_filt = filtered_emg
        # if count == 500:
        #     low_cut = 10
        #     high_cut = 450
        #     offline_emg = load("abd.bio")["emg"]
        #     EMG = Analogs(offline_emg)
        #     EMGBP = EMG.meca.band_pass(order=2, cutoff=[low_cut, high_cut], freq=2000)
        #     EMGBP_Centered = EMGBP.meca.center()
        #
        #     plt.figure()
        #     plt.plot(EMGBP_Centered.values[0, :10000])
        #     plt.plot(emg_filt[0, :])
        #     plt.show()

        # Save binary file
        # add_data_to_pickle({"emg": emg_tmp2}, output_file_path)

        count += 1
        loop_time = time() - tic
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(real_time_to_sleep)
