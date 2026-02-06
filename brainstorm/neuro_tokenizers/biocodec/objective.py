import torch, os
from .spectra import EEG2Spec, EMG2Spec


def total_loss(eeg_in, eeg_out, sr=250, is_emg=False, device="cuda"):
    """
    L_t: time loss | L_f: frequency loss
    Suggestion: lambda_t = 0.1 | lambda_f = 1
    """
    eeg_in = eeg_in.squeeze().to(device)
    eeg_out = eeg_out.squeeze().to(device)

    l1Loss = torch.nn.L1Loss(reduction="mean")
    l2Loss = torch.nn.MSELoss(reduction="mean")
    huber = torch.nn.SmoothL1Loss(reduction="mean", beta=0.7)

    # l_t: L1 distance between target and input EEG on time domain
    l_t = torch.tensor(0.0, device=device, requires_grad=True)
    l_t = huber(eeg_in, eeg_out)  # l1Loss(eeg_in, eeg_out)

    # l_f: L1+L2 over the computed STFT on several time scales
    l_f = torch.tensor(0.0, device=device, requires_grad=True)

    # channel weights: [logmag, cos(phi), sin(phi)]
    chan_w = torch.tensor([1.0, 0.2, 0.2], device=device).view(1, 3, 1, 1)
    low, high = (7, 11) if is_emg else (5, 9)
    for i in range(low, high):
        fft_fn = (
            EMG2Spec(
                n_fft=2**i,
                win_length=2**i,
                hop_length=(2**i) // 8,
                sampling_rate=sr,
                device=device,
            )
            if is_emg
            else EEG2Spec(
                n_fft=2**i,
                win_length=2**i,
                hop_length=(2**i) // 8,
                sampling_rate=sr,
                device=device,
            )
        )
        spec_in = fft_fn(eeg_in) * chan_w
        spec_out = fft_fn(eeg_out) * chan_w

        # Apply L1 + L2 (mean-normalized)
        l_f = l_f + l1Loss(spec_in, spec_out)
        l_f = l_f + l2Loss(spec_in, spec_out)

    # Pearson correlation coefficient per sample, then averaged
    def batch_corrcoef(x, y):
        x_mean = x.mean(dim=-1, keepdim=True)
        y_mean = y.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean
        numerator = (x_centered * y_centered).sum(dim=-1)
        denominator = x_centered.norm(dim=-1) * y_centered.norm(dim=-1) + 1e-8
        return (numerator / denominator).mean()

    corr = batch_corrcoef(eeg_in, eeg_out)

    # Scale-Invariant SDR
    def si_sdr(reference, estimate, eps=1e-8):
        reference = reference - reference.mean(dim=-1, keepdim=True)
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)

        scale = (reference * estimate).sum(dim=-1, keepdim=True) / (
            reference.pow(2).sum(dim=-1, keepdim=True) + eps
        )
        proj = scale * reference
        noise = estimate - proj

        ratio = proj.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + eps)
        return 10 * torch.log10(ratio + eps)

    sdr = si_sdr(eeg_in, eeg_out).mean()

    return {"l_t": l_t, "l_f": l_f, "corr": corr, "sdr": sdr}


if __name__ == "__main__":
    # Example usage
    eeg_in = torch.randn(16, 1250)
    eeg_out = torch.randn(16, 1250)
    loss = total_loss(eeg_in, eeg_out)
    print(loss)
