import torch
import torch.nn.functional as F

class HistogramLoss(torch.nn.Module):
    def __init__(self, bins=256):
        super(HistogramLoss, self).__init__()
        self.bins = bins

    def compute_histogram(self, img):
        """Compute histogram of an image."""
        histogram = torch.histc(img, bins=self.bins, min=0, max=1)
        # Normalize histogram
        histogram /= histogram.sum()
        return histogram

    def forward(self, img1, img2):
        """Compute histogram loss between two images."""
        # Split the channels and compute histograms
        hist1_red = self.compute_histogram(img1[:, 0, :, :])
        hist1_green = self.compute_histogram(img1[:, 1, :, :])

        hist2_red = self.compute_histogram(img2[:, 0, :, :])
        hist2_green = self.compute_histogram(img2[:, 1, :, :])

        # Compute loss for each channel
        loss_red = F.mse_loss(hist1_red, hist2_red)
        loss_green = F.mse_loss(hist1_green, hist2_green)

        # Combine losses from all channels
        loss = (loss_red + loss_green) / 3.0
        return loss

    