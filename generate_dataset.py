'''
Alex Knowlton
12/6/24

Generates constellation dataset for EECS453 and saves as .npy files
'''

import numpy as np
import matplotlib.pyplot as plt


def normalize_constellation(constellation):
    '''
    Normalizes such that each element in the constellation have energy 1
    '''
    E = np.sqrt(np.mean(constellation * np.conj(constellation)))
    Eb = E / np.log2(len(constellation))
    return constellation / Eb


def draw_samples(constellation, n_samples=512):
    '''
    draws n_samples from the given constellation
    '''
    M, n = constellation.shape
    indices = np.random.randint(0, M, n_samples)
    data = constellation[indices, :]
    real = np.real(data)
    imag = np.imag(data)
    data = np.vstack((real, imag))
    return data


def corrupt_data(samples, n0=0.05, n0_range=True, channel=1):
    '''
    Adds noise with standard deviation n0 to the given samples
    '''
    samples = samples.T.reshape((2, -1)).T
    samples = samples[:, 0] + 1j * samples[:, 1]
    samples = samples * channel
    samples = np.hstack((np.real(samples), np.imag(samples))).T
    if n0_range:
        n0 = np.random.uniform(0, n0)
    noise = np.random.normal(0, n0, samples.shape)
    data = samples + noise
    data = data.reshape((-1, 1))
    return data


def get_qam_constellation(n):
    '''
    Return QAM constellation with n points. Note: n must be an even power of 2
    '''
    b = int(np.sqrt(n)) - 1
    data = np.arange(-b, b+1, 2)
    N = len(data)
    real = data.reshape((1, N))
    imag = 1j * data.reshape((N, 1))
    data = real + imag
    L = data.shape[0] * data.shape[1]
    data = data.reshape((L, 1))
    return normalize_constellation(data)


def get_pam_constellation(n):
    '''
    Returns normalized PAM constellation with N points
    '''
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    points = np.exp(-1j * theta)
    points = points.reshape((points.shape[0], 1))
    return normalize_constellation(points)


def get_bpsk_constellation():
    return np.array([[-1], [1]])


def get_apsk32_constellation():
    inner = get_pam_constellation(4)
    middle = 3 * get_pam_constellation(12) * np.exp(-1j * 2 * np.pi / 24)
    outer = 5 * get_pam_constellation(16)
    constellation = np.vstack((inner, middle, outer))
    constellation = normalize_constellation(constellation)
    return constellation


def display_data(data, label):
    '''
    Plot data vector and display image
    '''
    M = data.shape[0] // 2
    real = data[:M]
    imag = data[M:]
    max_val = np.max(data) * 1.1
    plt.figure()
    plt.scatter(real, imag)
    plt.ylabel('Quadrature [Q]')
    plt.xlabel('In-phase [I]')
    plt.title(f'True Label: {label}')
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid()


def get_dataset(constellations, n_samples, n0=0.05, n0_range=True,
                channels=None):
    channels = [1] if channels is None else channels
    sample_length = 512
    results = np.zeros((sample_length * 2,
                        len(channels) * len(constellations) * n_samples))
    for j in range(len(constellations)):
        for i in range(len(channels)):
            print(f'Drawing sample {j},{i}')
            for k in range(n_samples):
                constellation = constellations[j]
                channel = channels[i]
                new_data = draw_samples(constellation)
                noisy_data = corrupt_data(new_data, n0, n0_range, channel)
                idx = k + j * n_samples + i * len(constellations)
                results[:, idx] = noisy_data[:, 0]
    outputs = np.array(range(len(constellations)))
    ones_list = np.ones((n_samples * len(channels), 1))
    outputs = outputs * ones_list
    outputs = outputs.T.reshape((outputs.shape[0] * outputs.shape[1]))
    outputs = outputs.astype(int)
    return results, outputs


def generate_dataset(n_samples, n0=0.05, n0_range=True,
                     channels=None, is_train=False):
    name = 'train' if is_train else 'test'

    constellations = []
    constellations.append(get_pam_constellation(4))
    constellations.append(get_pam_constellation(8))
    constellations.append(get_apsk32_constellation())
    constellations.append(get_bpsk_constellation())
    constellations.append(get_qam_constellation(4))
    constellations.append(get_qam_constellation(16))

    X, y = get_dataset(constellations,
                       n_samples=n_samples,
                       n0=n0, channels=channels, n0_range=n0_range)

    # reshape and form result arrays and save
    np.save(f'./data/{name}_results', y)
    np.save(f'./data/{name}_data', X.T)
    print('Finished Generating Dataset')


if __name__ == '__main__':
    generate_dataset(10, 5)
