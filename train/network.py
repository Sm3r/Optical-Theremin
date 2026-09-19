import torch
import torch.nn as nn


class HandNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        coord_mlp_dim: int = 128,
        hidden_dim: int = 48,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.coord_encoder = nn.Sequential(
            nn.Linear(input_dim, coord_mlp_dim),
            nn.SiLU(),
            nn.LayerNorm(coord_mlp_dim),
            nn.Dropout(dropout),
            nn.Linear(coord_mlp_dim, coord_mlp_dim),
            nn.SiLU(),
            nn.LayerNorm(coord_mlp_dim),
        )

        self.lstm = nn.LSTM(
            input_size=coord_mlp_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, state=None, return_state=False):
        z = self.coord_encoder(x)

        lstm_out, state = self.lstm(z, state)
        last = lstm_out[:, -1, :]

        y_hat = self.regressor(last).squeeze(-1)

        if return_state:
            return y_hat, state

        return y_hat


if __name__ == "__main__":

    ITERS = 1000
    INPUT_DIM = 18

    model = HandNet(input_dim=INPUT_DIM)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()

    x = torch.randn(1, 1, INPUT_DIM)

    with torch.no_grad():
        y_hat = model(x)
        print(y_hat.shape)

        import time

        start = time.time()
        for _ in range(ITERS):
            y_hat = model(x)
        end = time.time()

    elapsed = end - start
    avg_ms = elapsed * 1000.0 / ITERS

    print(f"{ITERS} iters: {elapsed:.2f} seconds")
    print(f"Average forward time: {avg_ms:.3f} ms")