import numpy as np

def get_raw_signals(index, data, time_length) :
  raw_signals = []
  for signals in data:
    raw_signals.append(signals[index][:time_length])
    
  return raw_signals

def get_signal_image(raw_signals):
  i = 1
  j = i + 1
  signal_sequences = 9
  signal_image = [raw_signals[i-1]]
  signal_index_string = str(i)
  
  while i != j:
    keys1 = str(i)+str(j)
    keys2 = str(j)+str(i)

    if j > signal_sequences:
      j = 1
    elif  (not keys1 in signal_index_string) and (not keys2 in signal_index_string):
      signal_image.append(raw_signals[j-1])
      signal_index_string += str(j)
      i = j
      j += 1  
    else:
      j += 1
  return signal_image

def get_activity_image(data, index):
  raw_signals = get_raw_signals(index, data, 68)
  signal_image = np.array(get_signal_image(raw_signals))
  dft_image = np.fft.fft2(signal_image, axes=(0, 1))
  dft_image_shifted = np.fft.fftshift(dft_image)
  magnitude = np.abs(dft_image_shifted)
  
  return magnitude