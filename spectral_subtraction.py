import scipy as sp
import scipy.io.wavfile as wav
import numpy as np

# Function for getting a single frame out of the signal
def get_frame(signal, winsize, index):
    shift = int(0.5 * winsize) # Shift between different frame windows.
    start = index * shift # 50% Overlap
    end = start + winsize
    return signal[start:end]

# Function for adding two signal frames. Used for overlap-and-add
def add_signal(signal, frame, winsize, index):
    shift = int(0.5 * winsize)
    start = index * shift
    end = start + winsize
    signal[start:end] = signal[start:end] + frame


def spec_sub(signal, noise, window):
    n_amp = sp.absolute(sp.fft(noise * window)) # Calculates noise frame amplitude
    s_spec = sp.fft(signal * window) 
    s_amp = sp.absolute(s_spec) # Calculates signal frame amplitude
    s_phase = sp.angle(s_spec) # Calculates signal frame phase

    out_amp = sp.sqrt((s_amp ** 2) - (n_amp ** 2)) # Performs spectral subtraction
    out_spec = out_amp * sp.exp(s_phase * 1j) # Reconstructs spectrum using noisy phase
    return sp.real(sp.ifft(out_spec)) # Returns ifft

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

# Filenames 
signal_filename = 'bar_10dB.wav'
noise_filename = 'barnoise.wav'
output_filename = 'bar_10dB_tratado_sub.wav'

winsize_ms = 25 # Window size in miliseconds
# Gain factor so that the noise in the noise recording has the same power as in the noisy signal
noise_gain = 0.1694240  

# Reading signal and noise files. Same sampling rate is assumed for both recordings
# Also, the files are assumed to be mono and not stereo 
(rate, signal) = wav.read(signal_filename)
wav.write('teste',rate,signal)
noise = wav.read(noise_filename)[1]
noise = noise[:len(signal)]
noise = noise * noise_gain
winsize = nextpow2(winsize_ms * 0.001 * rate) # Calculates winsize in samples
window = sp.hanning(winsize) # Calculates framing window

signal = np.pad(signal, (0,int((0.5 * winsize)-(len(signal) % (0.5 * winsize)))), 'constant') #Zero-padding
noise = np.pad(noise, (0,int((0.5 * winsize)-(len(noise) % (0.5 * winsize)))), 'constant')
out = sp.zeros(len(signal),np.int16) # Initializes output vector
number_frames = len(signal)/(0.5 * winsize) - 1 
for index in range(int(number_frames)): # Performs spectral subtraction for every frame
    s = get_frame(signal, winsize, index)
    n = get_frame(noise, winsize, index)
    add_signal(out, spec_sub(s, n, window), winsize, index) # And overlap-and-adds

wav.write(output_filename,rate,out) # Writes output .wav file