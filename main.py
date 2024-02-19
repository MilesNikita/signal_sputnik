import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSlider
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from scipy.signal import butter, filtfilt

class SuperheterodyneReceiver(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Супергетеродинный Приёмник")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.signal_frequency_label = QLabel("Частота сигнала")
        self.layout.addWidget(self.signal_frequency_label)
        self.signal_frequency_slider = QSlider(Qt.Horizontal)
        self.signal_frequency_slider.setMinimum(1)
        self.signal_frequency_slider.setMaximum(100)
        self.signal_frequency_slider.setValue(50)
        self.signal_frequency_slider.setTickPosition(QSlider.TicksBelow)
        self.signal_frequency_slider.setTickInterval(10)
        self.signal_frequency_slider.valueChanged.connect(self.plot_signal)
        self.layout.addWidget(self.signal_frequency_slider)
        self.heterodyne_frequency_label = QLabel("Частота гетеродина")
        self.layout.addWidget(self.heterodyne_frequency_label)
        self.heterodyne_frequency_slider = QSlider(Qt.Horizontal)
        self.heterodyne_frequency_slider.setMinimum(1)
        self.heterodyne_frequency_slider.setMaximum(100)
        self.heterodyne_frequency_slider.setValue(50)
        self.heterodyne_frequency_slider.setTickPosition(QSlider.TicksBelow)
        self.heterodyne_frequency_slider.setTickInterval(10)
        self.heterodyne_frequency_slider.valueChanged.connect(self.plot_signal)
        self.layout.addWidget(self.heterodyne_frequency_slider)
        self.noise_amplitude_label = QLabel("Амплитуда шума")
        self.layout.addWidget(self.noise_amplitude_label)
        self.noise_amplitude_slider = QSlider(Qt.Horizontal)
        self.noise_amplitude_slider.setMinimum(0)
        self.noise_amplitude_slider.setMaximum(100)
        self.noise_amplitude_slider.setValue(50)
        self.noise_amplitude_slider.setTickPosition(QSlider.TicksBelow)
        self.noise_amplitude_slider.setTickInterval(10)
        self.noise_amplitude_slider.valueChanged.connect(self.plot_signal)
        self.layout.addWidget(self.noise_amplitude_slider)
        self.intermediate_frequency_label = QLabel("Промежуточная частота")
        self.layout.addWidget(self.intermediate_frequency_label)
        self.intermediate_frequency_value = QLabel()
        self.layout.addWidget(self.intermediate_frequency_value)
        self.canvas_signal = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas_signal)
        self.toolbar_signal = NavigationToolbar(self.canvas_signal, self)
        self.layout.addWidget(self.toolbar_signal)
        self.canvas_mixed_signal = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas_mixed_signal)
        self.toolbar_mixed_signal = NavigationToolbar(self.canvas_mixed_signal, self)
        self.layout.addWidget(self.toolbar_mixed_signal)
        self.canvas_filtered_signal = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas_filtered_signal)
        self.toolbar_filtered_signal = NavigationToolbar(self.canvas_filtered_signal, self)
        self.layout.addWidget(self.toolbar_filtered_signal)
        self.plot_signal()

    def plot_signal(self):
        signal_freq = self.signal_frequency_slider.value()
        heterodyne_freq = self.heterodyne_frequency_slider.value()
        noise_amplitude = self.noise_amplitude_slider.value() / 100.0
        intermediate_freq = np.abs(signal_freq - heterodyne_freq) 
        time = np.linspace(0, 1, 500)
        signal = np.sin(2 * np.pi * signal_freq * time)
        heterodyne = np.sin(2 * np.pi * heterodyne_freq * time)
        mixed_signal = signal * heterodyne
        noise = np.random.normal(0, noise_amplitude, len(time))
        mixed_signal_with_noise = mixed_signal + noise
        filtered_signal = self.filter_signal(mixed_signal_with_noise)
        self.canvas_signal.figure.clear()
        self.canvas_mixed_signal.figure.clear()
        self.canvas_filtered_signal.figure.clear()
        ax1 = self.canvas_signal.figure.add_subplot(111)
        ax2 = self.canvas_mixed_signal.figure.add_subplot(111)
        ax3 = self.canvas_filtered_signal.figure.add_subplot(111)
        ax1.plot(time, signal, label='Исходный сигнал')
        ax2.plot(time, mixed_signal_with_noise, label='Смешанный сигнал с шумом')
        ax3.plot(time, filtered_signal, label='Отфильтрованный сигнал')
        ax1.set_xlabel('Время')
        ax1.set_ylabel('Амплитуда')
        ax1.set_title('Исходный сигнал')
        ax1.legend()
        ax1.grid(True)
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Амплитуда')
        ax2.set_title('Смешанный сигнал с шумом')
        ax2.legend()
        ax2.grid(True)
        ax3.set_xlabel('Время')
        ax3.set_ylabel('Амплитуда')
        ax3.set_title('Отфильтрованный сигнал')
        ax3.legend()
        ax3.grid(True)
        self.canvas_signal.draw()
        self.canvas_mixed_signal.draw()
        self.canvas_filtered_signal.draw()
        intermediate_freq_str = "{:.2f}".format(intermediate_freq)
        self.intermediate_frequency_value.setText(intermediate_freq_str)

    def filter_signal(self, signal):
        nyquist_freq = 0.5
        cutoff_freq = 0.2
        order = 2
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SuperheterodyneReceiver()
    window.show()
    sys.exit(app.exec_())
