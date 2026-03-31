# coding: utf-8
# SD-CDM: Strategy-Aware Dual-Channel Cognitive Diagnosis Model
# Paper: "SD-CDM: Strategy-Aware Dual-Channel Cognitive Diagnosis Model"
# Target journal: Expert Systems with Applications

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM


# ---------------------------------------------------------------------------
# CAM-MLP: two-layer MLP used inside the Cognitive Attention Machine
# ---------------------------------------------------------------------------
class CAM_MLP(nn.Module):
    """
    Two-layer MLP for computing raw attention scores (Eq. 6 in the paper).
    Input  : concatenated [v_habit || e_j || c_k]  →  shape (..., 3*hidden_dim)
    Output : scalar attention logit                →  shape (..., 1)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(torch.relu(self.W1(x)))


# ---------------------------------------------------------------------------
# Core network
# ---------------------------------------------------------------------------
class SDCDMNet(nn.Module):
    """
    Full forward-pass network of SD-CDM.

    Architecture (Section 3 of the paper):
        Stage 1 – Dual-Channel Knowledge Initialization  (BiKD module)
        Stage 2 – Strategy-Aware Knowledge Activation    (CAM  module)
        Stage 3 – Response Prediction
    """

    def __init__(self, knowledge_n: int, exer_n: int, student_n: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.knowledge_n = knowledge_n
        self.exer_n      = exer_n
        self.student_n   = student_n
        self.hidden_dim  = hidden_dim

        # ── BiKD Module ──────────────────────────────────────────────────
        # Dual-channel student trait matrices  E_pos, E_neg  ∈ R^{N×K}
        self.student_emb_pos = nn.Embedding(student_n, knowledge_n)   # positive mastery
        self.student_emb_neg = nn.Embedding(student_n, knowledge_n)   # negative misconception

        # Per-exercise difficulty scalar  d_j  and raw trap intensity  m_j
        self.e_difficulty    = nn.Embedding(exer_n, 1)
        self.trap_intensity  = nn.Embedding(exer_n, 1)  # m_j (sigmoid applied in forward)

        # ── CAM Module ───────────────────────────────────────────────────
        # Student habit vector  v^habit  ∈ R^D
        self.student_habit = nn.Embedding(student_n, hidden_dim)

        # Exercise and concept embeddings used to compute cross-feature attention
        self.exercise_emb  = nn.Embedding(exer_n,     hidden_dim)   # E_ex
        self.concept_emb   = nn.Embedding(knowledge_n, hidden_dim)  # E_con

        # CAM-MLP: input = 3D (habit + exercise + concept)
        self.cam_mlp = CAM_MLP(3 * hidden_dim, hidden_dim)

        # ── Weight Initialisation ─────────────────────────────────────────
        # Trap intensity prior: U(-2, -1)  →  initial μ_j ∈ (0.12, 0.27)
        # This prevents gradient vanishing in the negative channel early in training.
        nn.init.uniform_(self.trap_intensity.weight, -2.0, -1.0)

        # Xavier normal for all other weight matrices
        for name, param in self.named_parameters():
            if 'weight' in name and 'trap_intensity' not in name and param.dim() >= 2:
                nn.init.xavier_normal_(param)

    # ------------------------------------------------------------------ #
    def forward(self, stu_id: torch.Tensor,
                exer_id: torch.Tensor,
                knowledge_point: torch.Tensor):
        """
        Parameters
        ----------
        stu_id          : (B,)    student index
        exer_id         : (B,)    exercise index
        knowledge_point : (B, K)  Q-matrix row – binary mask of required concepts

        Returns
        -------
        pred            : (B,)    predicted P(r_ij = 1)
        h_pos           : (B, K)  positive mastery  h^+
        h_neg           : (B, K)  negative misconception  h^-
        knowledge_point : (B, K)  passed through so the caller can compute L_reg
        """
        B = stu_id.size(0)
        K = self.knowledge_n

        # ── Stage 1 : Dual-Channel Knowledge Initialisation (BiKD) ──────
        # Bounded trait vectors (Eq. 1)
        h_pos = torch.sigmoid(self.student_emb_pos(stu_id))   # (B, K)
        h_neg = torch.sigmoid(self.student_emb_neg(stu_id))   # (B, K)

        # Exercise parameters
        d_j  = torch.sigmoid(self.e_difficulty(exer_id)).squeeze(-1)   # (B,)
        mu_j = torch.sigmoid(self.trap_intensity(exer_id)).squeeze(-1) # (B,)  Eq. 2

        # ── Stage 2 : Strategy-Aware Knowledge Activation (CAM) ─────────
        v_habit = self.student_habit(stu_id)   # (B, D)
        e_j_emb = self.exercise_emb(exer_id)   # (B, D)
        c_k_all = self.concept_emb.weight       # (K, D)  shared table

        # Broadcast to (B, K, D) for vectorised attention computation
        v_h = v_habit.unsqueeze(1).expand(-1, K, -1)           # (B, K, D)
        e_j = e_j_emb.unsqueeze(1).expand(-1, K, -1)           # (B, K, D)
        c_k = c_k_all.unsqueeze(0).expand(B, -1, -1)           # (B, K, D)

        # Raw attention scores via CAM-MLP (Eq. 5–6)
        z_ijk = torch.cat([v_h, e_j, c_k], dim=-1)             # (B, K, 3D)
        a_ijk = self.cam_mlp(z_ijk).squeeze(-1)                 # (B, K)

        # Masked softmax: restrict attention to concepts required by this exercise
        # (Eq. 7 with Q-mask)
        mask    = (knowledge_point == 0)
        a_ijk   = a_ijk.masked_fill(mask, float('-inf'))
        alpha   = torch.softmax(a_ijk, dim=-1)                  # (B, K)
        alpha   = torch.nan_to_num(alpha, nan=0.0)              # guard all-masked rows

        # ── Stage 3 : Strategy-Weighted Prediction Fusion (Eq. 8) ───────
        # S̃_ij = Σ_{k∈K_j} α_ijk · (h+_ik  −  μ_j · h-_ik)  −  d_j
        dual_channel = h_pos - mu_j.unsqueeze(-1) * h_neg       # (B, K)
        S_ij = (alpha * dual_channel).sum(dim=-1) - d_j         # (B,)

        pred = torch.sigmoid(S_ij)                              # Eq. 9
        return pred, h_pos, h_neg, knowledge_point


# ---------------------------------------------------------------------------
# Public-facing CDM wrapper  (mirrors the NCDM API)
# ---------------------------------------------------------------------------
class SDCDM(CDM):
    """
    Strategy-Aware Dual-Channel Cognitive Diagnosis Model.

    Usage
    -----
    model = SDCDM(knowledge_n=110, exer_n=26660, student_n=4151)
    model.train(train_loader, test_loader, epoch=20, device='cuda')
    auc, acc = model.eval(test_loader, device='cuda')
    model.save('sdcdm.pt')
    """

    def __init__(self, knowledge_n: int, exer_n: int, student_n: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.sdcdm_net = SDCDMNet(knowledge_n, exer_n, student_n, hidden_dim)

    # ------------------------------------------------------------------ #
    def train(self, train_data, test_data=None, epoch: int = 20,
              device: str = "cpu", lr: float = 0.001,
              lam: float = 0.1, margin: float = 0.1,
              silence: bool = False):
        """
        Parameters
        ----------
        train_data : DataLoader yielding (user_id, item_id, knowledge_emb, y)
        test_data  : optional DataLoader for epoch-level evaluation
        epoch      : number of training epochs
        device     : 'cpu' or 'cuda'
        lr         : Adam learning rate
        lam        : weight of the margin regularisation term  λ  (Eq. 12)
        margin     : margin hyper-parameter  ε  in L_reg  (Eq. 11)
        silence    : suppress tqdm progress bars
        """
        self.sdcdm_net = self.sdcdm_net.to(device)
        bce = nn.BCELoss()
        # L2 regularisation is folded into Adam's weight_decay (λ_Θ in Eq. 12)
        optimizer = optim.Adam(self.sdcdm_net.parameters(),
                               lr=lr, weight_decay=1e-4)

        for epoch_i in range(epoch):
            self.sdcdm_net.train()
            epoch_losses = []

            for batch_data in tqdm(train_data,
                                   desc=f"Epoch {epoch_i}",
                                   disable=silence):
                user_id, item_id, knowledge_emb, y = batch_data
                user_id       = user_id.to(device)
                item_id       = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y             = y.to(device)

                pred, h_pos, h_neg, q = self.sdcdm_net(
                    user_id, item_id, knowledge_emb
                )

                # ── L_pred : Binary Cross-Entropy (Eq. 10) ───────────
                l_pred = bce(pred, y)

                # ── L_reg : Margin constraint on correct responses (Eq. 11)
                # For every correctly-answered item, positive mastery must
                # exceed misconception penalty by at least ε.
                correct_mask = (y == 1)
                if correct_mask.any():
                    pos_sum = (h_pos[correct_mask] * q[correct_mask]).sum(-1)
                    neg_sum = (h_neg[correct_mask] * q[correct_mask]).sum(-1)
                    l_reg   = torch.relu(neg_sum - pos_sum + margin).mean()
                else:
                    l_reg = torch.tensor(0.0, device=device)

                # ── Total loss (Eq. 12) ───────────────────────────────
                loss = l_pred + lam * l_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses))
            print(f"[Epoch {epoch_i}] average loss: {avg_loss:.6f}")

            if test_data is not None:
                auc, acc = self.eval(test_data, device=device)
                print(f"[Epoch {epoch_i}] auc: {auc:.6f}, accuracy: {acc:.6f}")

    # ------------------------------------------------------------------ #
    def eval(self, test_data, device: str = "cpu"):
        self.sdcdm_net = self.sdcdm_net.to(device)
        self.sdcdm_net.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch_data in tqdm(test_data, desc="Evaluating"):
                user_id, item_id, knowledge_emb, y = batch_data
                user_id       = user_id.to(device)
                item_id       = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)

                pred, _, _, _ = self.sdcdm_net(user_id, item_id, knowledge_emb)
                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(y.tolist())

        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
        return auc, acc

    # ------------------------------------------------------------------ #
    def save(self, filepath: str):
        torch.save(self.sdcdm_net.state_dict(), filepath)
        logging.info("Saved parameters to %s" % filepath)

    def load(self, filepath: str):
        self.sdcdm_net.load_state_dict(
            torch.load(filepath, map_location="cpu")
        )
        logging.info("Loaded parameters from %s" % filepath)

    # ------------------------------------------------------------------ #
    def get_student_profiles(self, student_ids: torch.Tensor, device: str = "cpu"):
        """
        Extract interpretable diagnostic profiles for a batch of students.

        Returns a dict with three keys matching Section 3.7 of the paper:
            'pos_mastery'   : (N, K) positive mastery  h^+
            'neg_misconception' : (N, K) negative misconception  h^-
            'habit_vector'  : (N, D) latent habit vector  v^habit

        These can be serialised as semantic priors for downstream LLMs.
        """
        self.sdcdm_net.eval()
        self.sdcdm_net = self.sdcdm_net.to(device)
        student_ids = student_ids.to(device)

        with torch.no_grad():
            h_pos   = torch.sigmoid(self.sdcdm_net.student_emb_pos(student_ids))
            h_neg   = torch.sigmoid(self.sdcdm_net.student_emb_neg(student_ids))
            v_habit = self.sdcdm_net.student_habit(student_ids)

        return {
            "pos_mastery":        h_pos.cpu(),
            "neg_misconception":  h_neg.cpu(),
            "habit_vector":       v_habit.cpu(),
        }
