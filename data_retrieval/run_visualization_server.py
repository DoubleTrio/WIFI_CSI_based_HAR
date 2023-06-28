# pylint: disable=import-error, missing-function-docstring, invalid-name, access-member-before-definition,
import argparse
import csv
import io
import os
from copy import deepcopy

import cv2
import numpy as np
from PyQt6 import QtNetwork, QtWidgets, uic
from pyqtgraph.Qt import QtCore

from csi_extraction import (
    calc_phase_angle,
    calibrate_amplitude,
    calibrate_phase,
    unpack_csi_struct,
)


class UI(QtWidgets.QWidget):
    def __init__(
        self,
        app: QtWidgets.QApplication,
        parent: QtWidgets.QWidget = None,
        is_5ghz: bool = True,
    ):
        super(UI, self).__init__(parent)
        self.app = app
        self.is_5ghz = is_5ghz

        # get and show object and layout
        uic.loadUi("window.ui", self)
        self.setWindowTitle(f"Visualize CSI data {'5GHz' if is_5ghz else '2.4GHz'}")

        self.antenna_pairs, self.carrier, self.amplitude, self.phase = [], [], [], []

        amp = self.box_amp
        amp.setBackground("w")
        amp.setWindowTitle("Amplitude")
        amp.setLabel("bottom", "Carrier", units="")
        amp.setLabel("left", "Amplitude", units="")
        amp.setYRange(0, 1, padding=0)  # for normalized amp values, prev range: 0, 90
        # amp.setYRange(0, 400, padding=0)  # for normalized amp values, prev range: 0, 90
        amp.setXRange(0, 114 if is_5ghz else 57, padding=0)

        def create_plot(field, color):
            return field.plot(pen={"color": color, "width": 3})
        
        self.colors = np.array([
            (200, 0, 0),
            (200, 200, 0),
            (137, 49, 239),
            (0, 0, 200),
            (0, 200, 200),
            (255, 0, 189),
            (135, 233, 17),
            (255, 127, 80),
            (0, 0, 0)
        ])

        self.penAmps = np.array(list(map(lambda x: create_plot(amp, x), self.colors))).reshape(3, 3)
        # self.penAmps[0].set
        # self.penAmp0_0 = create_plot(amp, (200, 200, 0))
        # print(self.penAmp0_0)
        # print("EHEHEHEHEEEaaaaaaaaaaaaaaaa")

        # self.penAmp0_1 = amp.plot(pen={"color": (200, 200, 0), "width": 3})
        # self.penAmp0_2 = amp.plot(pen={"color": (137, 49, 239), "width": 3})
        # self.penAmp1_0 = amp.plot(pen={"color": (0, 0, 200), "width": 3})
        # self.penAmp1_1 = amp.plot(pen={"color": (0, 200, 200), "width": 3})
        # self.penAmp1_2 = amp.plot(pen={"color": (255, 0, 189), "width": 3})
        # self.penAmp2_0 = amp.plot(pen={"color": (135, 233, 17), "width": 3})
        # self.penAmp2_1 = amp.plot(pen={"color": (255, 127, 80), "width": 3})
        # self.penAmp2_2 = amp.plot(pen={"color": (0, 0, 0), "width": 3})
        # self.ampList = []

        phase = self.box_phase
        phase.setBackground("w")
        phase.setWindowTitle("Phase")
        phase.setLabel("bottom", "Carrier", units="")
        phase.setLabel("left", "Phase", units="")
        phase.setYRange(-np.pi - 0.2, np.pi + 0.2, padding=0)
        phase.setXRange(0, 114 if is_5ghz else 57, padding=0)
        self.penPhases = np.array(list(map(lambda x: create_plot(phase, x), self.colors))).reshape(3, 3)

        self.penPhase0_1 = phase.plot(pen={"color": (200, 200, 0), "width": 3})
        self.penPhase0_2 = phase.plot(pen={"color": (137, 49, 239), "width": 3})
        self.penPhase1_0 = phase.plot(pen={"color": (0, 0, 200), "width": 3})
        self.penPhase1_1 = phase.plot(pen={"color": (0, 200, 200), "width": 3})
        self.penPhase1_2 = phase.plot(pen={"color": (255, 0, 189), "width": 3})
        self.penPhase2_0 = phase.plot(pen={"color": (135, 233, 17), "width": 3})
        self.penPhase2_1 = phase.plot(pen={"color": (255, 127, 80), "width": 3})
        self.penPhase2_2 = phase.plot(pen={"color": (0, 0, 0), "width": 3})

        self.amp = amp
        self.box_phase = phase

    @QtCore.pyqtSlot()
    def update_plots(self):
        # print(self.amplitude)
        # print(str(np.shape(self.amplitude)) + "HEREE")
        # print(np.shape(self.amplitude[:, 0, 0]))
        # print("HEREEE")

        # print(self.amplitude[:][0][0])

        # print(self.amplitude[0:55][0][0], )
        # print(self.amplitude.())
        if len(self.amplitude) and len(self.amplitude[0]):
            if len(self.antenna_pairs) > 0:
                # self.penAmps[0].setData(self.carrier, self.amplitude[:, 0, 0])
                # print(np.shape(self.amplitude)[0])
                for i in range(self.amplitude.shape[1]):
                    for j in range(self.amplitude.shape[2]):
                        self.penAmps[i, j].setData(self.carrier, self.amplitude[:, i, j])
                        self.penPhases[i, j].setData(self.carrier, self.phase[:, i, j])
                # self.penAmps[0].setData(self.carrier, self.phase[0])

                # if len(self.antenna_pairs) > 1:
                #     # self.penAmp0_1.setData(self.carrier, self.amplitude[1])
                #     self.penPhase0_1.setData(self.carrier, self.phase[1])

                # if len(self.antenna_pairs) > 2:
                #     # self.penAmp1_0.setData(self.carrier, self.amplitude[2])
                #     self.penPhase1_0.setData(self.carrier, self.phase[2])

                # if len(self.antenna_pairs) > 3:
                #     pass
                #     # self.penAmp1_1.setData(self.carrier, self.amplitude[3])
                #     self.penPhase1_1.setData(self.carrier, self.phase[3])

                # if len(self.antenna_pairs) > 4:
                #     # self.penAmp0_2.setData(self.carrier, self.amplitude[4])
                #     self.penPhase0_2.setData(self.carrier, self.phase[4])

                # if len(self.antenna_pairs) > 5:
                #     pass
                #     # self.penAmp1_2.setData(self.carrier, self.amplitude[5])
                #     self.penPhase1_2.setData(self.carrier, self.phase[5])

                # if len(self.antenna_pairs) > 6:
                #     pass
                #     # self.penAmp2_2.setData(self.carrier, self.amplitude[6])
                #     self.penPhase2_2.setData(self.carrier, self.phase[6])

                # if len(self.antenna_pairs) > 7:
                #     pass
                #     # self.penAmp2_1.setData(self.carrier, self.amplitude[7])
                #     self.penPhase2_1.setData(self.carrier, self.phase[7])

                # if len(self.antenna_pairs) > 8:
                #     pass
                #     # self.penAmp2_2.setData(self.carrier, self.amplitude[8])
                #     self.penPhase2_2.setData(self.carrier, self.phase[8])

        self.process_events()  # force complete redraw for every plot

    def process_events(self):
        self.app.processEvents()


class UDPListener:
    packet_counter = 0

    def __init__(
        self,
        save_data_path: str,
        sock: QtNetwork.QUdpSocket,
        form: UI,
        filename: str,
        make_photo: bool = False,
    ):
        self.save_data_path = save_data_path
        os.makedirs(save_data_path, exist_ok=True)
        self.filename = filename

        self.make_photo = make_photo
        self.cam = cv2.VideoCapture(0) if self.make_photo else None

        self.sock = sock
        sock.readyRead.connect(self.on_datagram_received)

        self.form = form

    def on_datagram_received(self):
        while self.sock.hasPendingDatagrams():
            print("Received new datagram")
            datagram, host, port = self.sock.readDatagram(
                self.sock.pendingDatagramSize()
            )
            f = io.BytesIO(datagram)
            csi_inf = unpack_csi_struct(f)

            if (
                csi_inf.csi != 0
            ):  # get csi from data packet, save and process for further visualization
                (
                    raw_peak_amplitudes,
                    raw_phases,
                    carriers_indexes,
                    antenna_pairs,
                    nam,
                    nph
                ) = self.get_csi_raw_data(csi_inf)

                if (len(raw_peak_amplitudes)):
                    self.calc(
                        nam, nph, carriers_indexes, antenna_pairs
                    )
                    self.save_csi_to_file(raw_peak_amplitudes, raw_phases, carriers_indexes)

                    if self.make_photo:
                        self.make_photo_and_save()

                    self.form.update_plots()
                    self.packet_counter += 1

    def get_csi_raw_data(self, csi_inf) -> np.ndarray | None:
        num_of_antenna_pairs = csi_inf.nc * csi_inf.nr

        print("getting csi from the raw data")


        channel = csi_inf.channel
        carriers_num = csi_inf.num_tones
        carriers = np.arange(csi_inf.num_tones)

        new_raw_phases = np.zeros([carriers_num, 3, 3])
        new_raw_amps = np.zeros([carriers_num, 3, 3])

        if (num_of_antenna_pairs < 9):
            print("WRONG NUMBER OF ANTENNAS...")
            # return [], None, None, None
            # # exit(1)
        print("channel: ", channel)
        print("carriers_num: ", carriers_num)

        antenna_pairs = [
            (tx_index, rx_index)
            for tx_index in range(csi_inf.nc)
            for rx_index in range(csi_inf.nr)
        ]
        raw_phases, raw_peak_amplitudes = [[] for _ in range(num_of_antenna_pairs)], [
            [] for _ in range(num_of_antenna_pairs)
        ]
        print("antenna_pairs: ", antenna_pairs)

        for i in range(carriers_num):
            for enum_index, (tr_i, rc_i) in enumerate(antenna_pairs):
                # print("csi_len ", csi_inf.csi_len)
                # print("csi_inf.csi[i]: ", csi_inf)
                p = csi_inf.csi[i][tr_i][rc_i]
                imag, real = p.imag, p.real

                peak_amplitude = np.sqrt(
                    np.power(real, 2) + np.power(imag, 2)
                )  # calculate peak amplitude
                phase_angle = calc_phase_angle(p)  # calculate phase angle

                raw_peak_amplitudes[enum_index].append(peak_amplitude)
                raw_phases[enum_index].append(phase_angle)

                new_raw_phases[i][tr_i][rc_i] = phase_angle
                new_raw_amps[i][tr_i][rc_i] = peak_amplitude
        # print(new_raw_amps)

        return raw_peak_amplitudes, raw_phases, carriers, antenna_pairs, new_raw_amps, new_raw_phases

    def calc(
        self, raw_peak_amplitudes, raw_phases, carriers_indexes, antenna_pairs
    ):  
        # Used for visualization only
        amplitude, phase = deepcopy(raw_peak_amplitudes), deepcopy(raw_phases)

        # Update form carriers indexes
        self.form.carrier = carriers_indexes

        # Calibrate amplitude and update form amplitude values
        amplitude = calibrate_amplitude(amplitude, 1)

        self.form.amplitude = amplitude

        # print(phase.shape)
        # Calibrate phase and update form phase values
        for i in range(phase.shape[1]):
            for j in range(phase.shape[2]):
                phase[:, i, j] = np.array(calibrate_phase(phase[:, i, j].flatten())).reshape(56)
        # phase = calibrate_phase(phase)
        # for i in range(len(phase)):
        #     print(len(phase[i]))
        #     phase[i] = calibrate_phase(phase[i])
        # phase = phase.reshape(3, 2)
        self.form.phase = phase
        self.form.antenna_pairs = antenna_pairs

    def make_photo_and_save(self):
        ret, frame = self.cam.read()

        img_name = "opencv_frame_{}.png".format(self.packet_counter)
        path_to_image = "{}/{}/{}".format(self.save_data_path, "images", img_name)

        cv2.imwrite(path_to_image, frame)
        print("{} written!".format(path_to_image))

    def save_csi_to_file(self, raw_peak_amplitudes, raw_phases, carriers):
        # print("Saving CSI data to .csv file...")
        filename_csv = self.filename
        path_to_csv_file = "{}/{}".format(self.save_data_path, filename_csv)

        with open(path_to_csv_file, "a", newline="\n") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(
                [
                    *carriers,
                    *raw_peak_amplitudes[0],
                    *raw_peak_amplitudes[1],
                    *raw_peak_amplitudes[2],
                    *raw_peak_amplitudes[3],
                    *raw_phases[0],
                    *raw_phases[1],
                    *raw_phases[2],
                    *raw_phases[3],
                ]
            )


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualization server that listens to any incoming packets, plot them and store.\n"
        "Only supports up to 4 antenna pairs at the moment.",
        prog="python run_visualization_server.py",
    )

    parser.add_argument(
        "-f",
        "--frequency",
        help="Frequency on which both routes are operating",
        choices=["2400MHZ", "5000MHZ"],
        default="2400MHZ",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        help="path to the folder where to save the incoming data",
        default="./tmp",
        type=str,
    )
    parser.add_argument(
        "-p", "--port", help="port to listen to", default=60000, type=int
    )
    parser.add_argument(
        "--photo",
        help="make webcam photo for each data packet?",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--file", help="name of the file to save to?", default="data.csv", type=str
    )

    return parser


def run_app() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    try:
        udp_socket = QtNetwork.QUdpSocket()
        udp_socket.bind(QtNetwork.QHostAddress.SpecialAddress.Any, args.port)

        form = UI(app=app, is_5ghz=(args.frequency == "5000MHZ"))
        form.show()

        listener = UDPListener(
            save_data_path=args.save_path,
            sock=udp_socket,
            form=form,
            filename=args.file,
            make_photo=args.photo,
        )
        app.exec()

    except KeyboardInterrupt as e:
        try:
            listener.cam.release()
        except Exception as e_cam:
            pass

        print("KeyboardInterrupt. Finishing the program...")


if __name__ == "__main__":
    # x = np.reshape(np.arange(56 * 3 * 3), [56, 3, 3])
    # print(np.shape(x))
    # print(x[:, 0, 0])
    run_app()
