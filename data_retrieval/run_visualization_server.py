# pylint: disable=import-error, missing-function-docstring, invalid-name, access-member-before-definition,
import argparse
import csv
import io
import os
from copy import deepcopy
import time
# import cv2
import numpy as np
from PyQt6 import QtNetwork, QtWidgets, uic

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import traceback, sys

from pyqtgraph.Qt import QtCore

cap = cv2.VideoCapture(0)

from csi_extraction import (
    calc_phase_angle,
    calibrate_amplitude,
    calibrate_phase,
    unpack_csi_struct,
)

def create_plot(field, color):
    return field.plot(pen={"color": color, "width": 3})

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class UI(QtWidgets.QWidget):
    
    def print_output(self, s):
        print(s)

    def print_finished(self):
        print("Finished recording!")

    def __init__(
        self,
        app: QtWidgets.QApplication,
        parent: QtWidgets.QWidget = None,
        is_5ghz: bool = True,
        make_video: bool = False,
        max_frames: int = 500
    ):
        super(UI, self).__init__(parent)
        self.make_video = make_video

        self.write_to_csv = True
        if (self.make_video):
            self.threadpool = QThreadPool()
            print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
            worker = Worker(self.useCamera)
            worker.signals.result.connect(self.print_output)
            worker.signals.finished.connect(self.print_finished)

            self.threadpool.start(worker)

        self.app = app
        self.count = 0
        self.max_frames = max_frames
        self.is_5ghz = is_5ghz

        # get and show object and layout
        uic.loadUi("window.ui", self)
        self.setWindowTitle(f"Visualize CSI data {'5GHz' if is_5ghz else '2.4GHz'}")

        amp = self.box_amp
        amp.setBackground("w")
        amp.setWindowTitle("Amplitude")
        amp.setLabel("bottom", "Carrier", units="")
        amp.setLabel("left", "Amplitude", units="")
        amp.setYRange(0, 1, padding=0)  # for normalized amp values, prev range: 0, 90
        # amp.setYRange(0, 400, padding=0)  # for normalized amp values, prev range: 0, 90
        amp.setXRange(0, 114 if is_5ghz else 57, padding=0)
        
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

        phase = self.box_phase
        phase.setBackground("w")
        phase.setWindowTitle("Phase")
        phase.setLabel("bottom", "Carrier", units="")
        phase.setLabel("left", "Phase", units="")
        phase.setYRange(-np.pi - 0.2, np.pi + 0.2, padding=0)
        phase.setXRange(0, 114 if is_5ghz else 57, padding=0)
        self.penPhases = np.array(list(map(lambda x: create_plot(phase, x), self.colors))).reshape(3, 3)
        self.amp = amp
        self.box_phase = phase

    @QtCore.pyqtSlot()
    def update_plots(self):
        self.count = self.count + 1
        print(self.count)
        if self.antenna_pairs:
            for i in range(self.amplitude.shape[1]):
                for j in range(self.amplitude.shape[2]):
                    self.penAmps[i, j].setData(self.carrier, self.amplitude[:, i, j])
                    self.penPhases[i, j].setData(self.carrier, self.phase[:, i, j])

        self.process_events()

    def process_events(self):
        self.app.processEvents()
    
    def useCamera(self):

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        # TODO remove the hardcode filename
        filename = 'video.mp4'

        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
        total_frames = 0
        while total_frames < self.max_frames:
            ret, frame= cap.read()

            writer.write(frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            print("Frame: " + str(total_frames))
            total_frames = total_frames + 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        # Stop writing to the CSV
        self.write_to_csv = False


class UDPListener:
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
        self.packet_counter = 0
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
                    raw_amplitudes,
                    raw_phases,
                    carriers_indexes,
                    antenna_pairs,
                ) = self.get_csi_raw_data(csi_inf)

                self.set_plot_data(
                    raw_amplitudes, raw_phases, carriers_indexes, antenna_pairs
                )

                # Only save 3x3 tensors
                if (len(antenna_pairs) == 3 * 3 and self.form.write_to_csv):
                    self.save_csi_to_file(csi_inf.num_tones, csi_inf.timestamp, raw_amplitudes, raw_phases)
                    if self.make_photo:
                        self.make_photo_and_save()

                # TODO, this plot more easily breaks at higher package sent 
                self.form.update_plots()
                self.packet_counter += 1

    def get_csi_raw_data(self, csi_inf) -> np.ndarray:
        print("Processing CSI Info")
        carriers_num = csi_inf.num_tones
        carriers = np.arange(csi_inf.num_tones)

        raw_amps = np.zeros([carriers_num, 3, 3])
        raw_phases = np.zeros([carriers_num, 3, 3])

        antenna_pairs = [
            (tx_index, rx_index)
            for tx_index in range(csi_inf.nc)
            for rx_index in range(csi_inf.nr)
        ]
        for i in range(carriers_num):
            for _, (tr_i, rc_i) in enumerate(antenna_pairs):
                p = csi_inf.csi[i][tr_i][rc_i]
                imag, real = p.imag, p.real

                peak_amplitude = np.sqrt(
                    np.power(real, 2) + np.power(imag, 2)
                )
                phase_angle = calc_phase_angle(p)

                raw_amps[i][tr_i][rc_i] = peak_amplitude
                raw_phases[i][tr_i][rc_i] = phase_angle

        return raw_amps, raw_phases, carriers, antenna_pairs

    def set_plot_data(
        self, raw_peak_amplitudes, raw_phases, carriers_indexes, antenna_pairs
    ):  
        # Used for visualization only
        amplitude, phase = deepcopy(raw_peak_amplitudes), deepcopy(raw_phases)

        # Update form carriers indexes
        self.form.carrier = carriers_indexes

        # Calibrate amplitude and update form amplitude values
        amplitude = calibrate_amplitude(amplitude, 1)

        self.form.amplitude = amplitude

        # Calibrate phase and update form phase values
        for i in range(phase.shape[1]):
            for j in range(phase.shape[2]):
                phase[:, i, j] = np.array(calibrate_phase(phase[:, i, j].flatten())).reshape(56)

        self.form.phase = phase
        self.form.antenna_pairs = antenna_pairs

    def make_photo_and_save(self):
        start = time.time()
        ret, frame = self.cam.read()
        

        img_name = "opencv_frame_{}.png".format(self.packet_counter)
        path_to_image = "{}/{}/{}".format(self.save_data_path, "images", img_name)

        cv2.imwrite(path_to_image, frame)
        end = time.time()
        print("{} written!".format(path_to_image))
        print("Execution time to take picture:", (end-start) * 10**3, "ms")

    def save_csi_to_file(self, num_carriers, timestamp, raw_amplitudes, raw_phases):
        # print("Saving CSI data to .csv file...")
        filename_csv = self.filename
        path_to_csv_file = "{}/{}".format(self.save_data_path, filename_csv)

        with open(path_to_csv_file, "a", newline="\n") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(
                [
                    num_carriers, timestamp, *raw_amplitudes.flatten(), *raw_phases.flatten()
                ]
            )

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualization server that listens to any incoming packets, plot them and store.\n"
        "Supports up to 9 antenna pairs at the moment.",
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
        help="Path to the folder where to save the incoming data",
        default="./tmp",
        type=str,
    )
    parser.add_argument(
        "-p", "--port", help="Port to listen to", default=60000, type=int
    )
    parser.add_argument(
        "--photo",
        help="Make webcam photo for each data packet?",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--video",
        help="Make a video until vp packets",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--file", help="Name of the file to save to?", default="data.csv", type=str
    )

    return parser


def run_app() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    try:
        udp_socket = QtNetwork.QUdpSocket()
        udp_socket.bind(QtNetwork.QHostAddress.SpecialAddress.Any, args.port)
        is_5ghz = args.frequency == "5000MHZ"
        make_video = args.video

        form = UI(app = app, is_5ghz = is_5ghz, make_video=make_video)
        form.show()

        listener = UDPListener(
            save_data_path=args.save_path,
            sock = udp_socket,
            form = form,
            filename = args.file,
            make_photo = args.photo,
        )
        app.exec()

    except KeyboardInterrupt as e:
        try:
            listener.cam.release()
        except Exception as e_cam:
            pass
        print("KeyboardInterrupt. Finishing the program...")


if __name__ == "__main__":
    print("Sleeping for 5 seconds... Get ready!")
    time.sleep(5)
    run_app()
