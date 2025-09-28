'''
Alex Knowlton
12/7/24

Generates plots for report
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
from generate_dataset import get_pam_constellation, get_qam_constellation, \
    draw_samples, corrupt_data, generate_dataset
from analytical_solve import get_mean_euclidean_distance
from train_model import train_model, test_model
from train_fading_model import train_fading_model, test_fading_model


def show_analytical_solution():
    '''
    Plots a visual aid for the analytical solution of the problem
    '''
    qam = get_qam_constellation(16)
    pam = get_pam_constellation(8)
    sample = draw_samples(pam, 128)
    sample = corrupt_data(sample, n0=0.25, n0_range=False)

    dE_pam = np.round(get_mean_euclidean_distance(sample, pam), 3)
    dE_qam = np.round(get_mean_euclidean_distance(sample, qam), 3)

    # reshape sample vector for plotting
    sample = sample.T.reshape((2, 128)).T
    sample = sample[:, 0] + 1j * sample[:, 1]

    # plot all and save image
    plt.figure(figsize=(5, 10))

    # plot PAM constellation with data
    plt.subplot(211)
    plt.title(f'PAM: $ d_E = {dE_pam} $')
    plt.scatter(np.real(sample), np.imag(sample), label='Sample Data')
    plt.scatter(np.real(pam), np.imag(pam), label='PAM Constellation')
    plt.grid()
    plt.legend(loc='upper right')
    plt.xlabel('In-phase [I]')
    plt.ylabel('Quadrature [Q]')

    # plot QAM constellation with data
    plt.subplot(212)
    plt.title(f'QAM: $ d_E = {dE_qam} $')
    plt.scatter(np.real(sample), np.imag(sample), label='Sample Data')
    plt.scatter(np.real(qam), np.imag(qam), label='QAM Constellation')
    plt.grid()
    plt.legend(loc='upper right')
    plt.xlabel('In-phase [I]')

    # save figure
    plt.tight_layout()
    plt.savefig('./img/analytical_comparison.png')


def plot_constellation_points():
    '''
    Plots 3 constellations on top of each other and save image
    '''
    qam4 = get_qam_constellation(4)
    qam16 = get_qam_constellation(16)
    pam = get_pam_constellation(8)

    # plot figure
    plt.figure()
    plt.grid()
    plt.scatter(np.real(qam16), np.imag(qam16), label='$QAM16$')
    plt.scatter(np.real(pam), np.imag(pam), label='$PAM$')
    plt.scatter(np.real(qam4), np.imag(qam4), label='$QAM4$', marker='+')
    plt.legend(loc='upper right')
    plt.xlabel('In-phase [I]')
    plt.ylabel('Quadrature [Q]')
    plt.title('Superimposed Constellations')
    plt.savefig('./img/constellation_compare.png')


def plot_constellation_with_channel():
    '''
    Plots a data sample with a constellation, but with a noisy flat fading
    channel, which looks rotated and shrunk
    '''
    qam = get_qam_constellation(16)
    data = draw_samples(qam)
    data = corrupt_data(data, n0=0.25, n0_range=False,
                        channel=0.4*np.exp(-1j * np.pi / 6))
    data = data.T.reshape((2, -1)).T
    data = data[:, 0] + data[:, 1] * 1j

    # plot figure
    plt.figure()
    plt.scatter(np.real(data), np.imag(data), label='Data')
    plt.scatter(np.real(qam), np.imag(qam), label='Constellation')
    plt.xlabel('In-phase [I]')
    plt.ylabel('Quadrature [Q]')
    plt.title('Data with Noisy Flat Fading Channel')
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig('./img/channel_fading.png')


def plot_constellation_with_noise():
    '''
    Plots and saves a QAM16 constellation with two different SNR values, and
    saves the image
    '''
    qam = get_qam_constellation(16)
    data = draw_samples(qam)
    data_low_snr = corrupt_data(data, n0=1.5, n0_range=False)
    data_low_snr = data_low_snr.reshape((2, -1)).T
    data_low_snr = data_low_snr[:, 0] + 1j * data_low_snr[:, 1]
    data_high_snr = corrupt_data(data, n0=0.5, n0_range=False)
    data_high_snr = data_high_snr.reshape((2, -1)).T
    data_high_snr = data_high_snr[:, 0] + 1j * data_high_snr[:, 1]

    # plot data
    plt.figure(figsize=(5, 10))
    plt.subplot(211)
    plt.scatter(np.real(data_low_snr), np.imag(data_low_snr))
    plt.scatter(np.real(qam), np.imag(qam))
    plt.title('Low SNR')
    plt.ylabel('Quadrature [Q]')
    plt.grid()

    plt.subplot(212)
    plt.scatter(np.real(data_high_snr), np.imag(data_high_snr), label='Data')
    plt.scatter(np.real(qam), np.imag(qam), label='Constellation')
    plt.title('High SNR')
    plt.ylabel('Quadrature [Q]')
    plt.xlabel('In-phase [I]')
    plt.legend(loc='upper right')
    plt.grid()

    plt.savefig('./img/constellation_with_noise.png')


def plot_correct_vs_noise(n_samples_train=5000, n_samples_test=500):
    '''
    Generates data with a signal to noise ratio and trains and tests the model
    to see how it behaves with more and more noisy data
    '''
    snr_db = np.linspace(-4, 1, num=10)
    n0_list = 1 / np.power(10, snr_db / 10)
    train_acc = []
    test_acc = []

    # generate dataset with base noise level
    generate_dataset(n_samples_train, 1.0,
                     n0_range=True, is_train=True)
    model = train_model()

    # test model with various noise levels
    for n0 in n0_list:
        print(f'Generating dataset for n0 = {n0}')
        generate_dataset(n_samples_test, n0, n0_range=False, is_train=False)
        e_test = test_model(net=model)
        test_acc.append(e_test)

    # plot output vs. sndr
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    plt.figure()
    plt.plot(snr_db, test_acc, label='Test Accuracy')
    plt.xlabel('$ E_b/N_0 $ [dB]')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs. SNR')
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig('./img/error_vs_snr.png')

    # save model
    dummy_input = torch.rand((1, 1, 2, 512), dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_input,),
        './model/CommsNet.onnx',
        input_names=['input']
    )


def plot_channel_correct_vs_noise(n_samples_train=1000, n_samples_test=100,
                                  channels=None):
    '''
    Generates data with a signal to noise ratio and trains and tests the model
    to see how it behaves with more and more noisy data
    '''
    snr_db = np.linspace(-4, 3, num=10)
    n0_list = 1 / np.power(10, snr_db / 10)
    test_acc = []

    # generate dataset
    generate_dataset(n_samples_train, 0.5,
                     n0_range=True, channels=channels, is_train=True)
    model = train_fading_model()

    for n0 in n0_list:
        print(f'Generating dataset for n0 = {n0}')
        generate_dataset(n_samples_test, n0, n0_range=False,
                         is_train=False)
        e_test = test_fading_model(net=model)
        test_acc.append(e_test)

    # plot output vs. sndr
    test_acc = np.array(test_acc)
    plt.figure()
    plt.plot(snr_db, test_acc)
    plt.xlabel('$ E_b/N_0 $ [dB]')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs. SNR with Fading Channel')
    plt.yscale('log')
    plt.grid()
    plt.savefig('./img/channel_error_vs_snr.png')


def main():
    show_analytical_solution()
    plot_constellation_points()
    plot_constellation_with_channel()
    plot_constellation_with_noise()
    plot_correct_vs_noise(5000)

    # amps = np.array([0.9, 0.7, 0.95, 1])
    # angles = 1j * np.pi * np.random.uniform(-1/4, 1/4, size=amps.shape)
    # angles[-1] = 1j * np.pi * 0
    # channels = amps * np.exp(angles)
    # plot_channel_correct_vs_noise(1000, channels=channels)


if __name__ == '__main__':
    main()
