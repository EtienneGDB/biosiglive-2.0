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
from heapq import nlargest
import scipy.io as sio

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
        .meca.low_pass(order=4, cutoff=3, freq=2000)
    )
    return np.array(pyo_signal_processed)

def threshold_activation_detection(device_data):
    mean_resting = np.array(np.mean(envelop(device_data), axis=1))
    mean_resting = mean_resting.reshape(n_electrodes, 1)
    std_resting = np.array(np.std(envelop(device_data), axis=1))
    std_resting = std_resting.reshape(n_electrodes, 1)
    thresh_muscle = mean_resting #+ 1*std_resting
    # thresh = np.array(np.mean(thresh_muscle))
    return thresh_muscle

def threshold_overactivation(device_data, window):
    MedmaxVal = np.median(nlargest(window, device_data))
    IQRmaxVal = stats.iqr(nlargest(window, device_data))
    thresh_overactivation = MedmaxVal + 1.5*IQRmaxVal
    return thresh_overactivation

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
        # interface2 = MyInterface(system_rate=100, data_path="abd.bio")
        interface = MyInterface(system_rate=100, data_path="Tri.bio")

        # Get prerecorded data from pickle file for a shoulder abduction
        # offline_emg = load("abd.bio")["emg"]
    else:
        # init trigno community client
        # interface = ViconClient(ip="localhost", system_rate=100)
        interface = PytrignoClient(system_rate=100, ip="127.0.0.1")


    # Add markerSet to Vicon interface
    # Muscles = ['Dent1', 'TrapInf', 'Bi', 'Tri', 'Dent2', 'DeltA', 'DeltM',
    #            'DeltP', 'TrapSup', 'TrapMed']
    # Muscles = ['DeltA', 'DeltM', 'DeltP']
    Muscles = ['Tri']
    muscle_names = Muscles
    n_electrodes = 1

    # -----SET PARAMETERS-----


    Rest_Act = True # Used to define resting activation level to detect periods of activation
    Calibration_Activation_level = True # Used to define an activation threshold to filter MVC or higher contractions
    nTime_Calib = 5
    # temp = np.zeros((n_electrodes, 0))
    # Baseline_emg_filtered = np.zeros((n_electrodes, nTime_Calib*1000))
    min_buffer = 5 # in seconds
    moving_window = 0.5 # in seconds
    Median_Frequency = np.zeros((n_electrodes, 50))
    Baseline_Median_Frequency = dict(zip(muscle_names, [np.array([])]*len(muscle_names)))
    Baseline_MF_Cond = np.zeros((n_electrodes, 1))
    # Evolution_Median_Frequency = np.zeros((n_electrodes, 0))
    Evolution_Median_Frequency = dict(zip(muscle_names, [np.array([])]*len(muscle_names)))
    dist_Activation_level = np.array([])
    MedmaxVal = np.zeros((n_electrodes, 1))
    IQRmaxVal = np.zeros((n_electrodes, 1))
    thresh_Activation_level = np.zeros((n_electrodes, 1))
    nSample_Baseline = 6
    Ev_MF = False
    nSample_Ev_MF = 6
    range_act_level = True
    IT = 1
    OverActivated = False
    # t_value = [[], [], [], []]
    # p_value = [[], [], [], []]

    output_file_path = "trial_x"

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

    # Order to give to participant
    print('Please completely rest muscles for 10 seconds for calibration purpose')
    # os.system("pause")

    # Add plot
    # emg_raw_plot = LivePlot(
    #     name="emg_raw", rate=100, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    # )
    # emg_raw_plot.init(plot_windows=1000, colors=(255, 0, 0), y_labels="EMG (mV)")

    emg_filtered_plot = LivePlot(
        name="emg_filtered", rate=100, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    # emg_filtered_plot.init(plot_windows=10000, y_labels="emg_filtered")

    emg_envelop = LivePlot(
        name="emg_envelop", rate=100, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    emg_envelop.init(plot_windows=10000, colors=(255, 0, 0), y_labels="emg_envelop")

    emg_medFreq = LivePlot(name="MedFreq", rate=10, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    emg_medFreq.init(plot_windows=50, y_labels="MedFreq")

    emg_thresh = LivePlot(name="Thresh", rate=10, plot_type=PlotType.Curve, nb_subplots=n_electrodes, channel_names=muscle_names
    )
    # emg_thresh.init(plot_windows=5000, y_labels="Thresh")

    time_to_sleep = 1 / 100
    count = 0
    # emg_filt=None
    # start = time()
    while True:
        # if count == 500:
        #     os.system("pause")
        tic = time()
        # mtn = datetime.datetime.now()
        raw_emg = interface.get_device_data(device_name="emg")
        raw_data = interface.devices[0].raw_data
        filtered_emg = interface.devices[0].process()
        # emg_raw_plot.update(raw_emg)
        # print(interface.devices[0].raw_data.shape)
        # emg_filtered_plot.update(filtered_emg[:, -20:])

        if filtered_emg[0][0] != 0:
            #-----DEFINE RESTING ACTIVATION LEVEL THRESHOLD FOR ACTIVATION DETECTION-----
            if Rest_Act:
                start = time()
                Resting_Activation_Threshold = RealTimeProcessing.custom_processing(
                    self=RealTimeProcessing, funct=threshold_activation_detection, data_tmp=filtered_emg
                )
                Rest_Act = False
                # os.system("pause")

            #-----CALCULATE ACTIVATION LEVEL-----
            Activation_level = RealTimeProcessing.custom_processing(
                self=RealTimeProcessing, funct=envelop, data_tmp=filtered_emg
            )

            #-----BUFFERING DATA-----
            if count < 101:
                temp_activation_level = Activation_level[:]
                temp_filtered_emg = filtered_emg[:]
            else:
                temp_activation_level = np.append(temp_activation_level, Activation_level[:, -20:], axis=1)
                temp_filtered_emg = np.append(temp_filtered_emg, filtered_emg[:, -20:], axis=1)

            #-----DYNAMIC THRESHOLD TO FILTER MVC-----
            if Calibration_Activation_level:
                if n_electrodes > 1:
                    Sum_Act_Level = np.sum(temp_activation_level, axis=0)
                else:
                    Sum_Act_Level = temp_activation_level
                if IT == 1:
                    dist_Activation_level = np.append(dist_Activation_level, Sum_Act_Level)
                    thresh_Activation_level = threshold_overactivation(dist_Activation_level, 5000)
                    IT = IT + 1
                elif (1 < IT < 403) and len(dist_Activation_level) < 10000:
                    dist_Activation_level = np.concatenate((dist_Activation_level, Sum_Act_Level[0, -20:]))
                    thresh_Activation_level = threshold_overactivation(dist_Activation_level, 10000)
                    IT = IT + 1
                elif IT > 402 and len(dist_Activation_level) < 10000:
                    dist_Activation_level = np.concatenate((dist_Activation_level, Sum_Act_Level[0, -20:]))
                else:
                    dist_Activation_level = np.concatenate((dist_Activation_level[20:], Sum_Act_Level[0, -20:]))
                    if sum(dist_Activation_level[-2000:] > thresh_Activation_level) > 500:
                        a = dist_Activation_level > thresh_Activation_level
                        a = a.astype(int)
                        b = np.diff(a)
                        c = np.array(np.where(b))
                        if c.size != 1:
                            c = c.squeeze()
                        cstart = c[range(0, c.size, 2)]
                        # cstart = cstart[cstart > 8000]
                        # cstop = c[range(1, c.size, 2)]
                        d = np.diff(np.insert(c, 0, 0))
                        e = d[range(1, d.size, 2)]

                        for i in range(len(e)):
                            if e[i] > 500:
                                print("Muscles over activated")
                                temp_activation_level = np.delete(temp_activation_level, np.s_[0:], axis=1)
                                temp_filtered_emg = np.delete(temp_filtered_emg, np.s_[0:], axis=1)
                                # dist_Activation_level = np.delete(dist_Activation_level, np.s_[
                                #             cstart[i]-round(len(dist_Activation_level)*0.02):cstop[i]]
                                #                                   )
                                dist_Activation_level = np.delete(dist_Activation_level, np.s_[
                                              cstart[i] - round(len(dist_Activation_level) * 0.05):])

                                thresh_Activation_level = threshold_overactivation(dist_Activation_level, 10000)

                                if Check_IT_overAct == Ref_IT_overAct:
                                    for iM in range(n_electrodes):
                                        Evolution_Median_Frequency[muscle_names[iM]] = np.delete(
                                            Evolution_Median_Frequency[muscle_names[iM]], -3)
                                        Check_IT_overAct = Check_IT_overAct + 1

                    New_thresh_Activation_level = threshold_overactivation(dist_Activation_level, 10000)
                    if (New_thresh_Activation_level > thresh_Activation_level) and (New_thresh_Activation_level < 1.5*thresh_Activation_level):
                        thresh_Activation_level = New_thresh_Activation_level
                        # print(thresh_Activation_level)

            # #-----VISUALIZE THRESHOLD-----
            # xxx = np.array([[thresh_Activation_level]])
            # emg_thresh.update(xxx[:, -1:])




            #-----WHEN ENOUGH DATA BUFFERED-----
            if temp_filtered_emg.shape[1] == min_buffer*2000:

                #-----Median Frequency calculation-----
                MF = RealTimeProcessing.custom_processing(
                    # self=RealTimeProcessing, funct=process_data_fFreqMed, data_tmp=temp_filtered_emg, interval=False, tps=mtn
                    self=RealTimeProcessing, funct=process_data_fFreqMed, data_tmp=temp_filtered_emg, interval=False
                )

                for iM in range(n_electrodes):
                    #-----CHECK IF MUSCLES ARE NOT OVERACTIVATED (MVC, OR LIFTING HEAVY CHARGE IN BETWEEN REPETITIVE TASK)-----
                    #---------------------CHECK HOW TO OPTIMIZE THRESHOLD------------------------------------------
                    # if any(temp_activation_level[iM, :] > thresh_Activation_level[iM, 1]*4):
                    #     print(muscle_names[iM], "over activated")
                    #     MF[iM] = 0
                    # else:
                    #-----SET MF TO "0" IF NOT ENOUGH ACTIVATION-----
                    if sum(temp_activation_level[iM, :] > Resting_Activation_Threshold[iM]) < min_buffer*2000/2:
                        MF[iM] = 0
                    else:
                        #-----DEFINE MEDIAN FREQUENCY BASELINE-----
                        if len(Baseline_Median_Frequency[muscle_names[iM]]) < nSample_Baseline:
                            Baseline_Median_Frequency[muscle_names[iM]] = np.append(Baseline_Median_Frequency[muscle_names[iM]], MF[iM])
                            Evolution_Median_Frequency[muscle_names[iM]] = Baseline_Median_Frequency[muscle_names[iM]]
                            Evolution_Median_Frequency[muscle_names[iM]] = np.delete(Evolution_Median_Frequency[muscle_names[iM]], -1)
                        else:
                            Baseline_MF_Cond[iM] = 1
                            #-----FILL EVOLUTION MEDIAN FREQUENCY-----
                            Evolution_Median_Frequency[muscle_names[iM]] = np.append(Evolution_Median_Frequency[muscle_names[iM]], MF[iM])
                            Ref_IT_overAct = 1
                            Check_IT_overAct = 1

                            #-----TELL WHEN MEDIAN FREQUENCY DECREASED BY X%-----
                            if Evolution_Median_Frequency[muscle_names[iM]].shape[0] == nSample_Ev_MF:
                                Mean_Baseline_MF = np.mean(Baseline_Median_Frequency[muscle_names[iM]])
                                Mean_Evolution_MF = np.mean(Evolution_Median_Frequency[muscle_names[iM]])
                                Percent_MF = Mean_Evolution_MF * 100 / Mean_Baseline_MF
                                Mean_Evolution_MF = np.array([[Mean_Evolution_MF]])
                                emg_medFreq.update(Mean_Evolution_MF[:, -1:])
                                if Percent_MF <= 95:
                                    print('Median frequency decreased by ', round(100-Percent_MF, 2), '%: please perform an MVC')
                                    Baseline_Median_Frequency[muscle_names[iM]] = Evolution_Median_Frequency[muscle_names[iM]]
                                Evolution_Median_Frequency[muscle_names[iM]] = np.delete(Evolution_Median_Frequency[muscle_names[iM]], -1)

                            # #-----TEST IF EVOLUTION MEDIAN FREQUENCY IS SIGNIFICANTLY LOWER THAN Baseline_Median_Frequency-----
                            # if Evolution_Median_Frequency[muscle_names[iM]].shape[0] == nSample_Ev_MF:
                            #     paired_t_test = stats.ttest_1samp(
                            #         Baseline_Median_Frequency[muscle_names[iM]],
                            #         np.mean(Evolution_Median_Frequency[muscle_names[iM]])
                            #     )
                            #     if (paired_t_test.pvalue < 0.05) & (Evolution_Median_Frequency[muscle_names[iM]].mean() -
                            #                                         Baseline_Median_Frequency[muscle_names[iM]].mean() < 0):
                            #         print(muscle_names[iM], "showed manifestation of muscle fatigue")
                            #         # Baseline_Median_Frequency[muscle_names[iM]] = Evolution_Median_Frequency[muscle_names[iM]]
                            #     Evolution_Median_Frequency[muscle_names[iM]] = np.delete(Evolution_Median_Frequency[muscle_names[iM]], -1)
                # emg_medFreq.update(MF[:, -1:])

                # emg_filtered_plot.update(temp_filtered_emg[:, -int(moving_window*2000):])
                emg_envelop.update(temp_activation_level[:, -int(moving_window*2000):])


                #-----REMOVE DATA BUFFERED FOR NEXT WINDOW BUFFERING-----
                temp_activation_level = np.delete(temp_activation_level, np.s_[:int(moving_window*2000)], axis=1)
                temp_filtered_emg = np.delete(temp_filtered_emg, np.s_[:int(moving_window*2000)], axis=1)

                # #-----CHECK IF MEDIAN FREQUENCY BASELINE IS FILLED-----
                # if sum(Baseline_MF_Cond) == n_electrodes:
                #     Ev_MF = True
                #     nSample_Baseline = 0

            # #-----FILL EVOLUTION MEDIAN FREQUENCY-----
            # if Ev_MF:
            #     Evolution_Median_Frequency = np.append(Evolution_Median_Frequency, MF, axis=1)


            # if Evolution_Median_Frequency.shape[1] == nSample_Ev_MF:
            #     for iM in range(n_electrodes):
            #         paired_t_test = stats.ttest_1samp(
            #             Baseline_Median_Frequency[muscle_names[iM]], np.mean(Evolution_Median_Frequency[iM])
            #         )
            #         if (paired_t_test.pvalue < 0.05) & (Evolution_Median_Frequency[iM, :].mean() -
            #                                             Baseline_Median_Frequency[iM, :].mean() < 0):
            #             print(muscle_names[iM], "showed manifestation of muscle fatigue")
            #             Baseline_Median_Frequency[iM] = Evolution_Median_Frequency[iM, :]
            #     Evolution_Median_Frequency = np.delete(Evolution_Median_Frequency, np.s_[:nSample_Ev_MF], axis=1)

            # if Ev_MF:
            #     Mean_Evolution_MF = Evolution_Median_Frequency[:, 0:].mean(axis=1)
            #     Mean_Evolution_MF = Mean_Evolution_MF.reshape(n_electrodes, 1)
            #     # Var_Evolution_MF = Evolution_Median_Frequency.var(axis=1)
            #     # Var_Evolution_MF = Var_Evolution_MF.reshape(4, 1)
            #     for iM in range(n_electrodes):
            #         paired_t_test = stats.ttest_1samp(Baseline_Median_Frequency[iM],
            #                                         np.mean(Evolution_Median_Frequency[iM]))
            #         # t_value[iM].append(paired_t_test.statistic)
            #         # p_value[iM].append(paired_t_test.pvalue)
            #         if (paired_t_test.pvalue < 0.05) & (Mean_Baseline_MF[iM] - Mean_Evolution_MF[iM] < 0):
            #             print(muscle_names[iM], "showed manifestation of muscle fatigue")
            #             Baseline_Median_Frequency = Evolution_Median_Frequency[:]
            #
            # if Baseline_Median_Frequency.shape[1] == nSample_Baseline:
            #     # Baseline_Median_Frequency = np.delete(Baseline_Median_Frequency, 0, axis=1)
            #     # Mean_Baseline_MF = Baseline_Median_Frequency[:, 0:].mean(axis=1)
            #     # Mean_Baseline_MF = Mean_Baseline_MF.reshape(n_electrodes, 1)
            #     # Var_Baseline_MF = Baseline_Median_Frequency.var(axis=1)
            #     # Var_Baseline_MF = Var_Baseline_MF.reshape(4, 1)
            #     nSample_Baseline = 0
            #     Ev_MF = True
            #     Evolution_Median_Frequency = Baseline_Median_Frequency[:, -nSample_Ev_MF:]

            # if Calibration_Activation_level:
            #     # Set dist_Activation_level
            #     if count == 100:
            #         dist_Activation_level = Activation_level[:]
            #         temp_filtered_emg = filtered_emg[:, :-20]
            #         temp_activation_level = Activation_level[:, :-20]
            #     if (count > 100) & (dist_Activation_level.shape[1] < nTime_Calib*2000):
            #         dist_Activation_level = np.append(dist_Activation_level, Activation_level[:, -20:], axis=1)
            #     if dist_Activation_level.shape[1] == nTime_Calib*2000:
            #         temp_filtered_emg = np.append(temp_filtered_emg, filtered_emg[:, -20:], axis=1)
            #         thresh_Activation_level = np.transpose(np.percentile(dist_Activation_level, [25, 75], axis=1))
            #         Baseline_thresh_emg_filtered = np.array([temp_filtered_emg[iM,
            #             (dist_Activation_level[iM, :] > thresh_Activation_level[iM, 0]) &
            #             (dist_Activation_level[iM, :] < thresh_Activation_level[iM, 1])] for iM in range(n_electrodes)])
            #         MF = RealTimeProcessing.custom_processing(
            #                 self=RealTimeProcessing, funct=process_data_fFreqMed, data_tmp=Baseline_thresh_emg_filtered, interval=False, tps=mtn
            #             )
            #         Baseline_Median_Frequency = np.append(Baseline_Median_Frequency, MF, axis=1)
            #         temp_filtered_emg = np.delete(temp_filtered_emg, np.s_[-20:], axis=1)
            #         temp_filtered_emg = np.delete(temp_filtered_emg, np.s_[:5000], axis=1)
            #         temp_activation_level = np.delete(temp_activation_level, np.s_[:5000], axis=1)
            #         Calibration_Activation_level = False

                # #-----DEFINE MEDIAN FREQUENCY BASELINE-----
                # if Baseline_Median_Frequency.shape[1] < nSample_Baseline:
                #     Baseline_Median_Frequency = np.append(Baseline_Median_Frequency, MF, axis=1)

            # emg_medFreq.update(MF)
            # Median_Frequency = np.append(Median_Frequency, MF, axis=1)
            # Median_Frequency = np.delete(Median_Frequency, 0, axis=1)
            # Mov_Av_MedFreq = np.mean(Median_Frequency, axis=1)
            # Mov_Av_MedFreq = Mov_Av_MedFreq[:, np.newaxis]

            # plt.figure()
            # for iM in range(n_electrodes):
            #     plt.subplot(2, 5, iM+1)
            #     plt.plot(temp_activation_level[iM, :])
            #     plt.plot([0, 5000], [Resting_Activation_Threshold[iM], Resting_Activation_Threshold[iM]])
            # plt.show()

            # emg_medFreq.update(Mov_Av_MedFreq[:, -1:])






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
        if loop_time > 0.01:
            print(loop_time)
        real_time_to_sleep = time_to_sleep - loop_time
        if real_time_to_sleep > 0:
            sleep(real_time_to_sleep)
