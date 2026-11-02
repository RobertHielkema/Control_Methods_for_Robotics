from lnn import UR7eLNN
import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# i could not ever complete a full run on my laptop so i used the best checkpoints
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = "ur7e_dynamics_dataset.npz"
OUT_DIR = "checkpoints/ur7e_lnn"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
WEIGHT_VELOCITY = 1.0
SEED = 42

def load_dataset(path=DATA_PATH):
    data = np.load(path)

    arrays = {
        "q": np.asarray(data["q"], dtype=np.float32),
        "qd": np.asarray(data["q_dot"], dtype=np.float32),
        "tau": np.asarray(data["action"], dtype=np.float32),
        "q_next": np.asarray(data["next_q"], dtype=np.float32),
        "qd_next": np.asarray(data["next_q_dot"], dtype=np.float32),
    }

    segment_id = np.asarray(data["segment_id"]).reshape(-1)

    n = len(arrays["q"])

    for key in ["q", "qd", "tau", "q_next", "qd_next"]:
        if arrays[key].shape != (n, 6):
            raise ValueError(
                f"{key}: expected shape ({n}, 6), "
                f"got {arrays[key].shape}"
            )
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains NaN or Inf.")

    if len(segment_id) != n:
        raise ValueError(
            f"segment_id has {len(segment_id)} rows, expected {n}"
        )
    dt = 0.001
    arrays["dt"] = np.full(n, dt, dtype=np.float32)

    print("Dataset loaded:")
    print(f"  Samples: {n}")
    print(f"  Segments: {len(np.unique(segment_id))}")
    print(f"  Timestep assumption: {dt:.8f} s")

    return arrays, segment_id


def split_by_segment(segment_id, seed=42):
    rng = np.random.default_rng(seed)
    unique_segments = rng.permutation(np.unique(segment_id))

    if len(unique_segments) < 3:
        raise ValueError(
            "Need at least 3 distinct segments for a "
            "train/validation/test split by segment."
        )

    n = len(unique_segments)
    n_train = max(1, int(0.70 * n))
    n_val = max(1, int(0.15 * n))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)

    train_ids = unique_segments[:n_train]
    val_ids = unique_segments[n_train:n_train + n_val]
    test_ids = unique_segments[n_train + n_val:]

    return (np.isin(segment_id, train_ids), np.isin(segment_id, val_ids), np.isin(segment_id, test_ids))

def tensors_for_mask(data, mask):
    return {
        k: torch.from_numpy(v[mask].copy())
        for k, v in data.items()
    }

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data = np.load("ur7e_dynamics_dataset.npz")

    q = data["q"]
    qd = data["q_dot"]
    action = data["action"]
    next_q = data["next_q"]
    next_qd = data["next_q_dot"]
    segment_id = data["segment_id"]

    dt = 0.001

    print("Shapes:")
    for key in data.files:
        print(f"  {key}: {data[key].shape}")
    print("\nUnique segments:", np.unique(segment_id).size)

    for name, x in [
        ("q", q),
        ("q_dot", qd),
        ("action", action),
        ("next_q", next_q),
        ("next_q_dot", next_qd),
    ]:
        print(
            f"{name:12s} min={x.min(axis=0)}\n"
            f"{'':12s} max={x.max(axis=0)}\n"
            f"{'':12s} std={x.std(axis=0)}"
        )
    position_residual = next_q - (q + qd * dt)
    velocity_change = (next_qd - qd) / dt

    print("\nMean |position transition residual| per joint:")
    print(np.mean(np.abs(position_residual), axis=0))

    print("\nEstimated acceleration range per joint:")
    print("min:", velocity_change.min(axis=0))
    print("max:", velocity_change.max(axis=0))

    print("\nMean absolute joint displacement per sample:")
    print(np.mean(np.abs(next_q - q)))

    print("\nAction standard deviation per joint:")
    print(action.std(axis=0))


    data, segment_id = load_dataset()
    train_mask, val_mask, test_mask = split_by_segment(segment_id)

    train = tensors_for_mask(data, train_mask)
    val = tensors_for_mask(data, val_mask)
    test = tensors_for_mask(data, test_mask)

    print(
        f"Samples: train={len(train['q'])}, "
        f"val={len(val['q'])}, test={len(test['q'])}"
    )

    q_mean = train["q"].mean(dim=0).numpy()
    q_std = train["q"].std(dim=0).clamp_min(0.05).numpy()

    q_scale = train["q"].std(dim=0).clamp_min(0.05).to(device)
    qd_scale = train["qd"].std(dim=0).clamp_min(0.1).to(device)

    model = UR7eLNN(q_mean, q_std).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-6
    )

    train_loader = DataLoader(
        TensorDataset(*train.values()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    keys = list(train.keys())

    def evaluate(split):
        model.eval()
        totals = []
        with torch.enable_grad():
            for start in range(0, len(split["q"]), BATCH_SIZE):
                batch = {
                    k: split[k][start:start + BATCH_SIZE].to(device)
                    for k in keys
                }
                q = batch["q"].detach().requires_grad_(True)
                qd = batch["qd"]
                tau = batch["tau"]

                q_pred, qd_pred, _ = model(
                    q, qd, tau, batch["dt"]
                )

                loss_q = (((q_pred - batch["q_next"]) / q_scale) ** 2).mean()
                loss_qd = (((qd_pred - batch["qd_next"]) / qd_scale) ** 2).mean()
                totals.append((loss_q + WEIGHT_VELOCITY * loss_qd).item())

        return float(np.mean(totals))

    best_val = float("inf")
    patience = 25
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for values in train_loader:
            batch = {
                k: v.to(device)
                for k, v in zip(keys, values)
            }

            q = batch["q"].detach().requires_grad_(True)
            qd = batch["qd"]
            tau = batch["tau"]

            q_pred, qd_pred, _ = model(q, qd, tau, batch["dt"])

            loss_q = (((q_pred - batch["q_next"]) / q_scale) ** 2).mean()
            loss_qd = (((qd_pred - batch["qd_next"]) / qd_scale) ** 2).mean()
            loss = loss_q + WEIGHT_VELOCITY * loss_qd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        val_loss = evaluate(val)

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "q_mean": q_mean,
                    "q_std": q_std,
                    "best_val_loss": best_val,
                },
                CHECKPOINT_DIR / "best_lnn.pt",
            )
        else:
            stale += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train={np.mean(train_losses):.6f} | "
                f"val={val_loss:.6f} | "
                f"best={best_val:.6f}"
            )

        if stale >= patience:
            print("Early stopping.")
            break

    checkpoint = torch.load(
        os.path.join(OUT_DIR, "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(test)
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Test loss: {test_loss:.6f}")

if __name__ == "__main__":
    main()