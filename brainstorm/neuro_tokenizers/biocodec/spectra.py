import torch, torch.nn as nn, os
import torch.nn.functional as F


class EEG2Spec(nn.Module):
    def __init__(
        self,
        n_fft=256,
        hop_length=32,
        win_length=256,
        sampling_rate=250,
        device="cuda",
    ):
        super().__init__()
        self.register_buffer(
            "window", torch.hann_window(win_length, device=device).float()
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate

    def forward(self, eeg_channel):
        # Zero-pad to handle STFT boundaries
        p = (self.n_fft - self.hop_length) // 2
        eeg_channel = F.pad(eeg_channel, (p, p), "reflect")

        # STFT (returns complex)
        stft_output = torch.stft(
            eeg_channel,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )

        # focus on magnitude information
        mag = stft_output.abs()
        logmag = torch.log1p(mag)

        # unit phasor
        phasor = stft_output / (mag.clamp_min(1e-8))
        return torch.stack((logmag, phasor.real, phasor.imag), dim=1)


class EMG2Spec(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_length=64,
        win_length=512,
        sampling_rate=1000,
        device="cuda",
    ):
        super().__init__()
        self.window = torch.hann_window(win_length, device=device).float()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate

    def forward(self, emg_channel):
        # Zero-pad to handle STFT boundaries
        p = (self.n_fft - self.hop_length) // 2
        emg_channel = F.pad(emg_channel, (p, p), "reflect")

        # STFT (returns complex)
        stft_output = torch.stft(
            emg_channel,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )

        # focus on magnitude information
        mag = stft_output.abs()
        logmag = torch.log1p(mag)

        # unit phasor
        phasor = stft_output / (mag.clamp_min(1e-8))
        return torch.stack((logmag, phasor.real, phasor.imag), dim=1)


if __name__ == "__main__":
    # Example parameters
    batch_size, sr = 4, 1000
    is_emg = sr == 1000
    seq_len = 5 * sr

    # Example signals
    eeg_input = torch.randn(batch_size, seq_len)
    eeg_input = eeg_input.cuda()

    # EEG range: [32, 256]
    # EMG range: [64, 512]
    low, high = (6, 10) if is_emg else (5, 9)
    for i in range(low, high):
        fft_fn = (
            EMG2Spec(
                n_fft=2**i,
                win_length=2**i,
                hop_length=(2**i) // 8,
                sampling_rate=sr,
                device="cuda",
            )
            if is_emg
            else EEG2Spec(
                n_fft=2**i,
                win_length=2**i,
                hop_length=(2**i) // 8,
                sampling_rate=sr,
                device="cuda",
            )
        )
        print("Output Shape:", fft_fn(eeg_input).shape)
