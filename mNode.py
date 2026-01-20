# pip install torch torchdiffeq pandas numpy scikit-learn scipy

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. Model Definition (mNODE)
# ==========================================

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, t, y):
        return self.net(y)


class mNODE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.ode_func = ODEFunc(hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = self.mlp1(x)
        t = torch.tensor([0.0, 1.0], device=x.device)
        h_t = odeint(self.ode_func, h0, t, method="dopri5")
        h1 = h_t[1]
        return self.mlp2(h1)


# ==========================================
# 2. Metrics and Loss (Julia-aligned)
# ==========================================

def mean_spearman_per_metabolite(y_pred: torch.Tensor, y_true: torch.Tensor, valid_mask=None) -> float:
    """
    Julia metric: Spearman per metabolite across samples, then mean.
    y_pred/y_true shape: (n_samples, n_metabolites)
    valid_mask: boolean array of length n_metabolites (optional)
    """
    yp = y_pred.detach().cpu().numpy()
    yt = y_true.detach().cpu().numpy()

    n_met = yt.shape[1]
    corrs = np.zeros(n_met, dtype=np.float64)

    for m in range(n_met):
        a = yp[:, m]
        b = yt[:, m]
        if np.std(a) == 0 or np.std(b) == 0:
            corrs[m] = 0.0
        else:
            corrs[m] = spearmanr(a, b).correlation

    if valid_mask is not None:
        corrs = corrs[valid_mask]

    return float(np.nanmean(corrs))


def l2_penalty(model: nn.Module) -> torch.Tensor:
    return sum((p ** 2).sum() for p in model.parameters())


# ==========================================
# 3. Training
# ==========================================

def train_mnode(
    X_train, y_train, X_val, y_val,
    hidden_dim: int,
    weight_decay: float,
    valid_mask=None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 5e-3,
    patience: int = 20
):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = mNODE(input_dim, hidden_dim, output_dim).to(device)

    # Julia adds L2 penalty inside the loss, so keep optimizer weight_decay=0 here
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    mse = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    best_scc = -1e9
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)

        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size: (i + 1) * batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)

            loss = mse(pred, yb) + weight_decay * l2_penalty(model)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            scc = mean_spearman_per_metabolite(val_pred, y_val_t, valid_mask=valid_mask)

        if scc > best_scc:
            best_scc = scc
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Val SCC: {scc:.4f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_scc, model


# ==========================================
# 4. Data loading (match Julia transpose)
# ==========================================

def load_data(path_prefix: str):
    X_train = pd.read_csv(os.path.join(path_prefix, "X_train.csv"), header=None).values
    y_train = pd.read_csv(os.path.join(path_prefix, "y_train.csv"), header=None).values
    X_test  = pd.read_csv(os.path.join(path_prefix, "X_test.csv"), header=None).values
    y_test  = pd.read_csv(os.path.join(path_prefix, "y_test.csv"), header=None).values

    # Julia uses readdlm(... )' so it transposes after loading.
    # If your saved CSVs were written for Julia, you MUST transpose here.
    X_train, y_train = X_train.T, y_train.T
    X_test,  y_test  = X_test.T,  y_test.T

    # compound names (tab-delimited in Julia)
    cn_path = os.path.join(path_prefix, "compound_names.csv")
    compound_names = None
    valid_mask = None
    if os.path.exists(cn_path):
        compound_names = pd.read_csv(cn_path, sep="\t", header=None).iloc[:, 0].astype(str).values
        # "annotated" ~= not empty and not NA-like
        valid_mask = np.array([c.strip() not in ("", "NA", "NaN", "nan", "None") for c in compound_names], dtype=bool)

    return X_train, y_train, X_test, y_test, compound_names, valid_mask


# ==========================================
# 5. Main: 5-fold CV + final training + save outputs
# ==========================================

if __name__ == "__main__":
    path_prefix = "./processed_data/"
    X_train, y_train, X_test, y_test, compound_names, valid_mask = load_data(path_prefix)

    print("=" * 50)
    print(f"Loaded: X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}")
    print("=" * 50)

    weight_decay_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    Nh_list = [32, 64, 128]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_avg_scc = -1e9
    best_params = (None, None)

    print("Begin hyperparameter selection:")
    for wd in weight_decay_list:
        for nh in Nh_list:
            fold_scores = []
            print(f"\nTesting: weight_decay={wd}, Nh={nh}")

            for train_idx, val_idx in kf.split(X_train):
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                X_va, y_va = X_train[val_idx], y_train[val_idx]

                scc, _ = train_mnode(
                    X_tr, y_tr, X_va, y_va,
                    hidden_dim=nh,
                    weight_decay=wd,
                    valid_mask=valid_mask,
                    epochs=50,          # CV shorter
                    batch_size=32,
                    learning_rate=5e-3,
                    patience=20
                )
                fold_scores.append(scc)

            avg_scc = float(np.mean(fold_scores))
            print(f"--> Average CV SCC: {avg_scc:.4f}")

            if avg_scc > best_avg_scc:
                best_avg_scc = avg_scc
                best_params = (wd, nh)

    best_wd, best_nh = best_params
    print("=" * 50)
    print(f"Selected Hyperparameters: weight_decay={best_wd}, Nh={best_nh}")
    print("=" * 50)

    print("Training final model on full training set...")
    final_scc, final_model = train_mnode(
        X_train, y_train, X_test, y_test,
        hidden_dim=best_nh,
        weight_decay=best_wd,
        valid_mask=valid_mask,
        epochs=100,
        batch_size=32,
        learning_rate=5e-3,
        patience=20
    )
    print(f"Final Test SCC: {final_scc:.4f}")

    os.makedirs("./results", exist_ok=True)
    final_model.eval()
    with torch.no_grad():
        preds = final_model(torch.tensor(X_test, dtype=torch.float32, device=device)).detach().cpu().numpy()

    np.savetxt("./results/predicted_metabolomic_profiles.csv", preds, delimiter=",")
    np.savetxt("./results/true_metabolomic_profiles.csv", y_test, delimiter=",")

    # metabolite-wise correlations like Julia (optional file)
    corrs = []
    for m in range(y_test.shape[1]):
        a = preds[:, m]
        b = y_test[:, m]
        if np.std(a) == 0 or np.std(b) == 0:
            corrs.append(0.0)
        else:
            corrs.append(spearmanr(a, b).correlation)
    np.savetxt("./results/metabolites_corr.csv", np.array(corrs), delimiter=",")
