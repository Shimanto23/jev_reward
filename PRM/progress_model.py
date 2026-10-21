import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressRewardModel(nn.Module):
    """Learn image -> task progress in [0, 1] from CoRe VLM scores."""

    def __init__(self, device, lr=1e-4, batch_size=32):
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(self.device)

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)

    def _images(self, images):
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[None]
        images = images.astype(np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))
        return torch.as_tensor(
            images, device=self.device, dtype=torch.float32
        )

    def predict(self, image):
        self.eval()
        with torch.no_grad():
            value = self(self._images(image)).item()
        self.train()
        return float(value)

    def predict_batch(self, images, batch_size=64):
        self.eval()
        values = []

        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                x = self._images(images[start:start + batch_size])
                values.append(self(x).squeeze(-1).cpu().numpy())

        self.train()
        return (
            np.concatenate(values)
            if values
            else np.empty((0,), dtype=np.float32)
        )

    def train_from_rrm(self, rrm, updates=5):
        if rrm.size_segment <= 1:
            raise ValueError(
                "PRM requires segment > 1 to use CoRe frame-progress scores."
            )

        size = rrm.capacity if rrm.buffer_full else rrm.buffer_index
        if size <= 0:
            return 0.0

        x = np.concatenate([
            rrm.buffer_seg1[:size],
            rrm.buffer_seg2[:size]
        ], axis=0)

        y = np.concatenate([
            rrm.buffer_score_1[:size],
            rrm.buffer_score_2[:size]
        ], axis=0)

        x = x.reshape(-1, x.shape[2], x.shape[3], 3)
        y = y.reshape(-1).astype(np.float32)

        valid = np.isfinite(y)
        x = x[valid]
        y = np.clip(y[valid], 0.0, 1.0)

        if len(y) == 0:
            return 0.0

        losses = []

        for _ in range(int(updates)):
            order = np.random.permutation(len(y))

            for start in range(0, len(y), self.batch_size):
                idx = order[start:start + self.batch_size]
                images = self._images(x[idx])
                targets = torch.as_tensor(
                    y[idx],
                    device=self.device,
                    dtype=torch.float32
                ).view(-1, 1)

                pred = self(images)
                loss = F.smooth_l1_loss(pred, targets)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                losses.append(loss.item())

        torch.cuda.empty_cache()
        return float(np.mean(losses)) if losses else 0.0

    def save(self, model_dir, step):
        torch.save(
            self.state_dict(),
            os.path.join(model_dir, f"progress_model_{step}.pt")
        )
