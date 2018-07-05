import scipy as sp
import scipy.io.wavfile as wav
import numpy as np
import os

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

def wiener(signal, noise, window):
    n_amp = sp.absolute(sp.fft(noise * window)) # Calculates noise frame amplitude
    s_spec = sp.fft(signal * window) 
    s_amp = sp.absolute(s_spec) # Calculates signal frame amplitude
    s_phase = sp.angle(s_spec) # Calculates signal frame phase

    out_spec = s_spec * ((s_amp ** 2) - 0.7*(n_amp ** 2))/(s_amp ** 2) # Performs wiener filtering
    return sp.real(sp.ifft(out_spec)) # Returns ifft

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

# Filenames 
signal_filename = './Audios/Edited/Fala-Separado-2.wav'
noise_filename = './Audios/Edited/Ruido Separado 2.wav'
output_filename = './Audios/Treated/Fala-Separado-2-Tratado.wav'

winsize_ms = 25 # Window size in miliseconds

# Reading signal and noise files. Same sampling rate is assumed for both recordings
# Also, the files are assumed to be mono and not stereo 
(rate, signal) = wav.read(signal_filename)
noise = wav.read(noise_filename)[1]
noise = noise[:len(signal)]
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

# Calls Watson Speech-To-Text via bash
# Command string
cmd_string_begin = 'curl -X POST -u 15813c96-9241-449e-a630-13adec4ce1fd:cS14woNs4nmt \
--header "Content-Type: audio/wav" \
--data-binary @'
cmd_string_end =' \
"https://stream.watsonplatform.net/speech-to-text/api/v1/recognize?model=pt-BR_BroadbandModel"'
# Bash Command
os.system(cmd_string_begin + signal_filename + cmd_string_end)
os.system(cmd_string_begin + output_filename + cmd_string_end)
