"""
Vanishing Gradients in Vanilla RNNs
====================================
A self-contained demo for a computational modelling course.

Two parts:
  Part 1 — Gradient autopsy: directly measure how gradients shrink
           as we backprop through time steps.
  Part 2 — Real failure: train a vanilla RNN on a long-range memory
           task and watch it fail, then compare to an LSTM.

Requirements: pip install torch matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim

# ─── reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ════════════════════════════════════════════════════════════════════════════
# PART 1 — GRADIENT AUTOPSY
# ════════════════════════════════════════════════════════════════════════════
#
# Idea: run a forward pass on a sequence of length T, compute a scalar loss
# at the END, then backprop and record ‖∂Loss/∂h_t‖ at every timestep t.
#
# If gradients vanish, the early timesteps will have near-zero gradient,
# meaning the network is blind to anything that happened far in the past.
# ─────────────────────────────────────────────────────────────────────────────

def gradient_autopsy(T=50, hidden_size=32, input_size=1):
    """
    Forward pass of length T, then backprop.
    Returns gradient norms at each timestep.
    """
    rnn_cell = nn.RNNCell(input_size, hidden_size, nonlinearity='tanh')

    # Random input sequence — we don't care about the task here,
    # just the gradient flow.
    x = torch.randn(T, input_size)
    h = torch.zeros(hidden_size, requires_grad=False)

    # Store hidden states so we can call .retain_grad() on each
    hiddens = []
    for t in range(T):
        h = rnn_cell(x[t].unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        h.retain_grad()          # ← normally intermediate grads are discarded
        hiddens.append(h)

    # Scalar loss at the very last timestep only
    loss = hiddens[-1].sum()
    loss.backward()

    # Collect ‖∂loss/∂h_t‖ for each t
    grad_norms = [h.grad.norm().item() for h in hiddens]
    return grad_norms


print("Part 1: Running gradient autopsy …")
T = 60
grad_norms = gradient_autopsy(T=T)

# ─── also vary sequence length to show the trend ────────────────────────────
lengths = [10, 20, 40, 80]
grad_at_first_step = []
for length in lengths:
    norms = gradient_autopsy(T=length)
    grad_at_first_step.append(norms[0])   # gradient at t=0 (the oldest step)


# ════════════════════════════════════════════════════════════════════════════
# PART 2 — LONG-RANGE MEMORY TASK
# ════════════════════════════════════════════════════════════════════════════
#
# Task: "Echo the first token after a long silence"
#
# Input:  [signal, 0, 0, 0, …, 0]    (T tokens total)
# Target: predict the signal value at the LAST step
#
# The network must remember what it saw at t=0 across T-1 filler steps.
# Short sequences → easy. Long sequences → vanilla RNN forgets.
# ─────────────────────────────────────────────────────────────────────────────

def make_batch(batch_size, seq_len, num_classes=8):
    """
    Returns (inputs, targets).
    inputs:  (batch, seq_len, 1) — one-hot class encoded as float in [0,1]
    targets: (batch,)            — the class shown at t=0
    """
    labels = torch.randint(0, num_classes, (batch_size,))
    # encode label as a unique value in (0,1) so it's a regression target
    signal = (labels.float() + 1) / (num_classes + 1)   # avoids 0 and 1

    inputs = torch.zeros(batch_size, seq_len, 1)
    inputs[:, 0, 0] = signal          # only the first timestep carries info
    targets = signal                   # predict it at the last timestep
    return inputs, targets


class VanillaRNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size,
                          batch_first=True, nonlinearity='tanh')
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)   # last timestep → scalar


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train(model, seq_len, epochs=300, batch_size=128, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        x, y = make_batch(batch_size, seq_len)
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


# Train both models on short and long sequences
print("Part 2: Training models …")

seq_short, seq_long = 10, 50
results = {}

for label, seq_len in [("short (T=10)", seq_short), ("long  (T=50)", seq_long)]:
    rnn_model  = VanillaRNN(hidden_size=64)
    lstm_model = LSTMModel(hidden_size=64)
    results[label] = {
        "rnn":  train(rnn_model,  seq_len),
        "lstm": train(lstm_model, seq_len),
    }
    print(f"  {label} done")


# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(15, 10))
fig.suptitle("Vanishing Gradients in Vanilla RNNs", fontsize=15, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {"rnn": "#e05c5c", "lstm": "#4a90d9"}

# ── Plot 1: gradient norms along time (the autopsy) ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
timesteps = list(range(T))
ax1.semilogy(timesteps, grad_norms, color='#e05c5c', lw=2)
ax1.axhline(1e-6, color='gray', ls='--', lw=1, label='Effective zero')
ax1.fill_between(timesteps, grad_norms, alpha=0.15, color='#e05c5c')
ax1.set_title("‖∂Loss/∂h_t‖ across a single backprop pass  (T=60, log scale)",
              fontsize=11)
ax1.set_xlabel("Timestep t  →  (t=59 is where loss is computed)")
ax1.set_ylabel("Gradient norm (log)")
ax1.legend(fontsize=9)
ax1.annotate("Network is\n'blind' here",
             xy=(5, grad_norms[5]), xytext=(15, grad_norms[5]*50),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

# ── Plot 2: gradient at t=0 vs sequence length ────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.semilogy(lengths, grad_at_first_step, 'o-', color='#e05c5c', lw=2, ms=8)
ax2.set_title("Gradient at t=0\nvs sequence length", fontsize=11)
ax2.set_xlabel("Sequence length T")
ax2.set_ylabel("‖∂Loss/∂h_0‖ (log)")
for x, y in zip(lengths, grad_at_first_step):
    ax2.annotate(f"{y:.1e}", (x, y), textcoords="offset points",
                 xytext=(5, 5), fontsize=8)

# ── Plots 3 & 4: training loss curves ────────────────────────────────────────
for col, (label, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, col + (1 if col == 1 else 0)])
    ax.plot(res["rnn"],  label="Vanilla RNN", color=COLORS["rnn"],  lw=1.5)
    ax.plot(res["lstm"], label="LSTM",        color=COLORS["lstm"], lw=1.5)
    ax.set_title(f"Memory task — {label}", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

# Re-lay the 3rd plot cleanly (it shares a cell with plot 2 area workaround)
ax_long = fig.add_subplot(gs[1, 2])
res = results["long  (T=50)"]
ax_long.plot(res["rnn"],  label="Vanilla RNN", color=COLORS["rnn"],  lw=1.5)
ax_long.plot(res["lstm"], label="LSTM",        color=COLORS["lstm"], lw=1.5)
ax_long.set_title("Memory task — long (T=50)", fontsize=11)
ax_long.set_xlabel("Epoch")
ax_long.set_ylabel("MSE Loss")
ax_long.legend(fontsize=9)
ax_long.set_ylim(bottom=0)

# Fix: overwrite the duplicate axis created in the loop
fig.axes[3].remove()   # remove the wrongly placed 4th axis from the loop

plt.savefig("/mnt/user-data/outputs/vanishing_gradients.png", dpi=150, bbox_inches='tight')
print("\nSaved plot → vanishing_gradients.png")
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY (useful to print in a notebook cell)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nGradient at the FIRST timestep (T=60 sequence):")
print(f"  ‖∂Loss/∂h_0‖ = {grad_norms[0]:.2e}   ← essentially zero")
print(f"  ‖∂Loss/∂h_59‖ = {grad_norms[-1]:.2e}  ← where loss lives\n")

for label, res in results.items():
    rnn_final  = np.mean(res["rnn"][-20:])
    lstm_final = np.mean(res["lstm"][-20:])
    print(f"Memory task [{label}]")
    print(f"  Vanilla RNN final loss : {rnn_final:.4f}")
    print(f"  LSTM        final loss : {lstm_final:.4f}")
    winner = "LSTM wins 🏆" if lstm_final < rnn_final * 0.7 else "similar"
    print(f"  → {winner}\n")