import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_fid import fid_score, inception
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import mne

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = "cuda" if torch.cuda.is_available() else "cpu"


"""
Plot samples
"""
def plot_samples(samples, n, text):
    fig, axs = plt.subplots(n, 1, figsize=(6,8))
    i = 0
    for _ in range(n):
        rnd_idx = np.random.choice(len(samples))
        s = samples[rnd_idx]
        axs[i].plot(s)
        i += 1

    fig.suptitle(f"{text} Samples (Scaled)", fontsize = 16)
    fig.tight_layout()
    plt.show()

"""
Compute FID_score of generated samples
"""
def fid_score(signaldir, samplesdir, device):
    paths = [signaldir, samplesdir]
    fid_score = fid_score.calculate_fid_given_paths(paths, batch_size=61, device=device, dims = 2048, num_workers=0)
    print("FID score:", fid_score)

"""
Compute average Wasserstein distance and minimal Euclidean distance of generated samples
"""
def wd_and_ed_score(dataimport, samples, N):
    all_wd = []
    all_ed = []
    time = 0
    for i in dataimport:
        for sample in samples[4999, :, 0, :]:
            all_wd.append(wasserstein_distance(i, sample))
            all_ed.append(distance.euclidean(i, sample))
        time += 1
        print('step', time, '/', N)
    print("Mean Wasserstein Distance:", np.mean(all_wd))
    print("Minimal Euclidean Distance:", np.min(all_ed))

"""
Generate a power spectral density plot
"""
def psd_plot(samples, vae_samples):
    fmin = 1.
    fmax = 100.
    dif_all_psds = []
    dif_all_freqs = []
    for i in range(350):
        psds_open, freqs_open = mne.time_frequency.psd_array_welch(samples[4999][i][0], sfreq = 256, fmin=fmin, fmax=fmax, window='boxcar', verbose=False)
        dif_all_psds.append(psds_open)
        dif_all_freqs.append(freqs_open)

    vae_all_psds = []
    vae_all_freqs = []
    for sample in vae_samples[:, :, 0]:
        psds_open, freqs_open = mne.time_frequency.psd_array_welch(sample, sfreq = 256, fmin=fmin, fmax=fmax, window='boxcar', verbose=False)
        vae_all_psds.append(psds_open)
        vae_all_freqs.append(freqs_open)

    plt.figure(figsize=(14, 4))
    plt.plot(np.mean(dif_all_freqs, axis=0), 10 * np.log10(np.mean(dif_all_psds, axis=0)), label="Diffusion model", alpha=0.6)
    plt.plot(np.mean(vae_all_freqs, axis=0), 10 * np.log10(np.mean(vae_all_psds, axis=0)), label="TimeVAE", alpha=0.6)
    plt.legend()
    plt.title("Combined PSD spectra of samples from TIMEVAE and the diffusion model")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [dB]')
    plt.show()

"""
Perform evaluation functions
"""
if __name__ == 'evaluate_models':
    datadir = "" #directory of dataset
    signaldir = "" #directory of real signal images
    samplesdir = "" #directory of sample signal images
    dataimport = np.load(datadir)
    N, T = dataimport.shape

    samples = np.load("") #directory of generated samples (by diffusion model or TimeVAE)
    vae_samples = np.load("") #directory of generated samples by TimeVAE

    fid_score(signaldir, samplesdir, device)
    wd_and_ed_score(dataimport, samples, N)
    psd_plot(samples, vae_samples)
