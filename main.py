import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSlider, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from scipy.signal import butter, filtfilt, welch

class SuperheterodyneReceiver(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Супергетеродинный Приёмник")
        self.setGeometry(100, 100, 1200, 600)  # Increased width for accommodating frequency plots
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
        self.open_img = QPushButton('Показать схему супергетеродинного приёмника')
        self.open_img.clicked.connect(self.show_img)
        self.layout.addWidget(self.open_img)        
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
        self.clear_subplots()
        self.plot_time_domain_signals(time, signal, mixed_signal_with_noise, filtered_signal)
        self.plot_frequency_domain_signals(signal, mixed_signal_with_noise, filtered_signal)
        self.draw_canvases(intermediate_freq)

    def clear_subplots(self):
        self.canvas_signal.figure.clear()
        self.canvas_mixed_signal.figure.clear()
        self.canvas_filtered_signal.figure.clear()

    def plot_time_domain_signals(self, time, signal, mixed_signal_with_noise, filtered_signal):
        ax1 = self.canvas_signal.figure.add_subplot(231)
        ax2 = self.canvas_mixed_signal.figure.add_subplot(231)
        ax3 = self.canvas_filtered_signal.figure.add_subplot(231)
        ax1.plot(time, signal, label='Исходный сигнал')
        ax2.plot(time, mixed_signal_with_noise, label='Смешанный сигнал с шумом')
        ax3.plot(time, filtered_signal, label='Отфильтрованный сигнал')
        self.set_common_properties(ax1, 'Исходный сигнал')
        self.set_common_properties(ax2, 'Смешанный сигнал с шумом')
        self.set_common_properties(ax3, 'Отфильтрованный сигнал')

    def plot_frequency_domain_signals(self, signal, mixed_signal_with_noise, filtered_signal):
        f_signal, Pxx_signal = welch(signal, fs=500)
        f_mixed_signal, Pxx_mixed_signal = welch(mixed_signal_with_noise, fs=500)
        f_filtered_signal, Pxx_filtered_signal = welch(filtered_signal, fs=500)
        ax4 = self.canvas_signal.figure.add_subplot(233)
        ax5 = self.canvas_mixed_signal.figure.add_subplot(233)
        ax6 = self.canvas_filtered_signal.figure.add_subplot(233)
        ax4.semilogy(f_signal, Pxx_signal)
        ax5.semilogy(f_mixed_signal, Pxx_mixed_signal)
        ax6.semilogy(f_filtered_signal, Pxx_filtered_signal)
        self.set_common_properties(ax4, 'Спектр исходного сигнала')
        self.set_common_properties(ax5, 'Спектр смешанного сигнала с шумом')
        self.set_common_properties(ax6, 'Спектр отфильтрованного сигнала')

    def set_common_properties(self, ax, title):
        ax.set_xlabel('Частота (Гц)' if title.startswith('Спектр') else 'Время')
        ax.set_ylabel('Амплитуда' if title.startswith('Спектр') else 'Амплитуда')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    def draw_canvases(self, intermediate_freq):
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
    
    def show_img(self):
        image_window = QWidget()
        image_window.setWindowTitle("Cхема супергетеродинного приёмника")
        image_label = QLabel(image_window)
        pixmap = QPixmap("scale_1200.png")  
        image_label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(image_label)
        image_window.setLayout(layout)
        image_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SuperheterodyneReceiver()
    window.show()
    sys.exit(app.exec_())
