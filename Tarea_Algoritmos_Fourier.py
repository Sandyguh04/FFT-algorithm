#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import wavfile
import numpy as np

fs, data = wavfile.read('song1.wav')
print("Frecuencia de muestreo: ", fs, "Hz")
print("Número de muestras: ", len(data))
print("Duración del archivo: ", len(data)/fs, "segundos")
print("Tipo de dato: ", data.dtype)


# In[2]:


timeArray = np.arange(0, 1024*16*16*2, 1)
timeArray = timeArray / fs
timeArray = timeArray * 1000  #scale to milliseconds


# In[3]:


import matplotlib
import matplotlib.pyplot as plt

s1 = data[:,0] 
plt.plot(timeArray, s1[:1024*16*16*2], color='k')
plt.ylabel('Amplitud')
plt.xlabel('Tiempo [ms]')
plt.show()


# In[4]:


s2 = data[:,1]
plt.plot(timeArray, s2[:1024*16*16*2], color='k')
plt.ylabel('Amplitud')
plt.xlabel('Tiempo [ms]')
plt.show()


# In[5]:


def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype = float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


# In[6]:


def FFT(x):
    """implementación recursiva 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("tamaño de x debe ser potencia de 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[0::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])


# In[7]:


def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()


# In[8]:


#x = np.random.random(1024 * 16)
x = s1[:1024*16*16*2]
#print(len(x))
get_ipython().run_line_magic('timeit', 'FFT(x)')
get_ipython().run_line_magic('timeit', 'FFT_vectorized(x)')
get_ipython().run_line_magic('timeit', 'np.fft.fft(x)')

i1 = FFT(x)
i2 = FFT_vectorized(x)
i3 = np.fft.fft(x)


# In[30]:


import math

n = 1024*16*16*2 
nUniquePts = int(math.ceil((n+1) / 2.0))
i3_g1 = i2[0:nUniquePts]
i3_g = abs(i3_g1)


# In[31]:


i3_g = i3_g / float(n) # scale by the number of points so that
                 # the magnitude does not depend on the length 
                 # of the signal or on its sampling frequency  
i3_g = i3_g**2  # square it to get the power 

# multiply by two
# odd nfft excludes Nyquist point
if n % 2 > 0: # we've got odd number of points fft
    i3_g[1:len(i3_g)] = i3_g[1:len(i3_g)] * 2
else:
    i3_g[1:len(i3_g) -1] = i3_g[1:len(i3_g) - 1] * 2 # we've got even number of points fft

freqArray = np.arange(0, nUniquePts, 1.0) * (fs / n);
plt.plot(freqArray / 1000, 10 * np.log10(i3_g), color = 'k')
plt.xlabel('Frecuencia (kHz)')
plt.ylabel('Energía (dB)')
plt.show()


# In[32]:


rms_val = math.sqrt(np.mean(s1**2))
print(rms_val)


# In[33]:


print(math.sqrt(sum(i3_g)))


# In[44]:


s = np.fft.ifft(i3_g)
timeArray = np.arange(0, len(s), 1)
timeArray = timeArray / fs
timeArray = timeArray * 1000  #scale to milliseconds
plt.plot(timeArray, s.real, 'b-', timeArray, s.imag, 'r--')
plt.legend(('real', 'imaginario'))
plt.show()


# In[ ]:





# In[ ]:




