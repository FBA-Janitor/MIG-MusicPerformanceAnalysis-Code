import math
import numpy as np
import scipy
import librosa

class FeatureExtractor(object):
    """
    Input:
        - x: np.adarray, (audio_len, )
    """
    def __init__(self, block_size=4096, hop_size=2048) -> None:

        self.block_size = block_size
        self.hop_size = hop_size

    def __call__(self, x, sr):
        # Compute the spectrogram (freq, time)
        # TODO: don't use scipy, use librosa instead
        hops = np.arange(0, len(x) -1, self.hop_size)
        hops += self.block_size
        big = max(hops[hops < len(x) - 1])
        num_iterations = int((big - self.block_size) / self.hop_size) + 1
        _, _, s = scipy.signal.spectrogram(x, nperseg=self.block_size, noverlap=self.hop_size, mode='magnitude')
        
        # Extract the feature
        # TODO: Extract the feature with packages, and fix the zero
        feature = np.vstack([
            self._time_rms(x, sr)[:num_iterations],
            self._spectral_crest_factor(s, sr)[np.newaxis][:num_iterations],
            self._spectral_centroid(s, sr)[:num_iterations],
            self._time_zero_crossing_rate(x, sr)[:num_iterations],
            self._spectral_rolloff(s, sr)[:num_iterations],
            self._spectral_flux(s, sr)[np.newaxis][:num_iterations],
            self._spectral_mfcc(s, sr)[:num_iterations],
            np.zeros(num_iterations, dtype=np.float32)
        ])
 

        # Arrange the features in larger blocks
        # where each large block contains the 
        # mean and std of the small frames,
        # and the larger blocks ~ 1s
        # The output will be like
        # (num_seconds, num_feature * 2)
        num_frame_combine = int(sr / self.hop_size) - 1
        f_bin, t_bin = feature.shape
        
        # FIXME: big bug here!!
        # The commented should be the correct
        ################
        # feature = feature[:, :num_frame_combine * (t_bin // num_frame_combine)].reshape(
        #     f_bin, t_bin // num_frame_combine, num_frame_combine)
        # feature = np.concatenate([feature.mean(axis=-1), feature.std(axis=-1)], axis=0).T
        #################

        
        # This is a graceful reimplementation of the bug
        feature = feature[::2]
        feature = feature[:, :num_frame_combine * (t_bin // num_frame_combine)].reshape(
            (f_bin + 1) // 2, t_bin // num_frame_combine, num_frame_combine)
        feature = np.concatenate([feature.mean(axis=-1), feature.std(axis=-1)], axis=1).reshape(
            (f_bin + 1) // 2 * 2, t_bin // num_frame_combine)
        feature = np.pad(feature, [(0, f_bin * 2 - feature.shape[0]), (0, 0)])  # fill the feature with zeros lol
        return feature.T

    def _spectral_centroid(self, spec, sr):

        # X = X**2 removed for consistency with book

        norm = spec.sum(axis=0, keepdims=True)
        norm[norm == 0] = 1

        vsc = np.dot(np.arange(0, spec.shape[0]), spec) / norm

        # convert from index to Hz
        vsc = vsc / (spec.shape[0] - 1) * sr / 2

        return np.squeeze(vsc)

    def _spectral_crest_factor(self, spec, sr):

        norm = spec.sum(axis=0)
        if np.ndim(spec) == 1:
            if norm == 0:
                norm = 1
        else:
            norm[norm == 0] = 1

        vtsc = spec.max(axis=0) / norm
        return vtsc

    def _spectral_flux(self, spec, sr):
        # difference spectrum (set first diff to zero)
        spec = np.c_[spec[:, 0], spec]

        afDeltaX = np.diff(spec, 1, axis=1)
        # flux
        vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / spec.shape[0]

        return vsf

    def _time_rms(self, audio, sr):
        # number of results
        iNumOfBlocks = math.ceil(audio.size / self.hop_size)

        # compute time stamps
        t = (np.arange(0, iNumOfBlocks) * self.hop_size + (self.block_size / 2)) / sr

        # allocate memory
        vrms = np.zeros(iNumOfBlocks)

        for n in range(0, iNumOfBlocks):

            i_start = n * self.hop_size
            i_stop = np.min([audio.size - 1, i_start + self.block_size - 1])

            # calculate the rms
            vrms[n] = np.sqrt(np.dot(audio[np.arange(i_start, i_stop + 1)], audio[np.arange(i_start, i_stop + 1)]) / (i_stop + 1 - i_start))

        # convert to dB
        epsilon = 1e-5  # -100dB

        vrms[vrms < epsilon] = epsilon
        vrms = 20 * np.log10(vrms)

        return vrms

    def _time_zero_crossing_rate(self, audio, sr):
        # number of results
        iNumOfBlocks = math.ceil(audio.size / self.hop_size)

        # compute time stamps
        t = (np.arange(0, iNumOfBlocks) * self.hop_size + (self.block_size / 2)) / sr

        # allocate memory
        vzc = np.zeros(iNumOfBlocks)

        for n in range(0, iNumOfBlocks):

            i_start = n * self.hop_size
            i_stop = np.min([audio.size - 1, i_start + self.block_size - 1])

            # calculate the zero crossing rate
            vzc[n] = 0.5 * np.mean(np.abs(np.diff(np.sign(audio[np.arange(i_start, i_stop + 1)]))))

        return vzc

    def _spectral_rolloff(self, spec, sr, kappa=0.85):
            
        spec = np.cumsum(spec, axis=0) / spec.sum(axis=0, keepdims=True)

        vsr = np.argmax(spec >= kappa, axis=0)

        # convert from index to Hz
        vsr = vsr / (spec.shape[0] - 1) * sr / 2

        return vsr

    def _spectral_mfcc(self, spec, sr, n_mfcc=24):
        
        # allocate memory
        v_mfcc = np.zeros([n_mfcc, spec.shape[1]])

        # generate filter matrix
        H = ToolMfccFbL(spec.shape[0], sr)
        T = generateDctMatrixL(H.shape[0], n_mfcc)

        for n in range(0, spec.shape[1]):
            # compute the mel spectrum
            X_Mel = np.log10(np.dot(H, spec[:, n] + 1e-20))

            # calculate the mfccs
            v_mfcc[:, n] = np.dot(T, X_Mel)

        return (v_mfcc)


# see function mfcc.m from Slaneys Auditory Toolbox
def generateDctMatrixL(iNumBands, iNumCepstralCoeffs):
    T = np.cos(np.outer(np.arange(0, iNumCepstralCoeffs), (2 * np.arange(0, iNumBands) + 1)) * np.pi / 2 / iNumBands)

    T = T / np.sqrt(iNumBands / 2)
    T[0, :] = T[0, :] / np.sqrt(2)

    return (T)


def ToolMfccFbL(iFftLength, f_s):

    # initialization
    f_start = 133.3333

    iNumLinFilters = 13
    iNumLogFilters = 27
    iNumFilters = iNumLinFilters + iNumLogFilters

    linearSpacing = 66.66666666
    logSpacing = 1.0711703

    # compute band frequencies
    f = np.zeros(iNumFilters + 2)
    f[np.arange(0, iNumLinFilters)] = f_start + np.arange(0, iNumLinFilters) * linearSpacing
    f[np.arange(iNumLinFilters, iNumFilters + 2)] = f[iNumLinFilters - 1] * logSpacing**np.arange(1, iNumLogFilters + 3)

    # sanity check
    if f[iNumLinFilters - 1] >= f_s / 2:
        f = f[f < f_s / 2]
        iNumFilters = f.shape[0] - 2

    f_l = f[np.arange(0, iNumFilters)]
    f_c = f[np.arange(1, iNumFilters + 1)]
    f_u = f[np.arange(2, iNumFilters + 2)]

    # allocate memory for filters and set max amplitude
    H = np.zeros([iNumFilters, iFftLength])
    afFilterMax = 2 / (f_u - f_l)
    f_k = np.arange(0, iFftLength) / (iFftLength - 1) * f_s / 2

    # compute the transfer functions
    for c in range(0, iNumFilters):
        # lower filter slope
        i_l = np.argmax(f_k > f_l[c])
        i_u = np.max([0, np.argmin(f_k <= f_c[c]) - 1])
        H[c, np.arange(i_l, i_u + 1)] = afFilterMax[c] * (f_k[np.arange(i_l, i_u + 1)] - f_l[c]) / (f_c[c] - f_l[c])
        # upper filter slope
        i_l = i_u + 1
        i_u = np.max([0, np.argmin(f_k < f_u[c]) - 1])
        H[c, np.arange(i_l, i_u + 1)] = afFilterMax[c] * (f_u[c] - f_k[np.arange(i_l, i_u + 1)]) / (f_u[c] - f_c[c])

    return (H)


if __name__ == '__main__':
    x, _ = librosa.load("/home/yding402/fba-data/test_test_audio/29701/29701.mp3")
    extractor = FeatureExtractor(block_size=4096, hop_size=2048)
    print(extractor(x, sr=22050).shape)

    
