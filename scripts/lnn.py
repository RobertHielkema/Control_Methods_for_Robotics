import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class UR7eLNN(nn.Module):

    def __init__(self, q_mean, q_std, hidden=128, epsilon=1e-3,):
        super().__init__()
        self.n = 6
        self.epsilon = epsilon

        self.register_buffer(
            "q_mean", torch.as_tensor(q_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "q_std",
            torch.as_tensor(q_std, dtype=torch.float32).clamp_min(1e-3)
        )
        self.mass_net = MLP(6, 21, hidden)
        self.potential_net = MLP(6, 1, hidden)
        self.raw_damping = nn.Parameter(torch.full((6,), -2.0))

        idx = torch.tril_indices(6, 6)
        self.register_buffer("tri_i", idx[0])
        self.register_buffer("tri_j", idx[1])
        self.register_buffer(
            "diag_mask", idx[0] == idx[1]
        )

    def normalized_q(self, q):
        return (q - self.q_mean) / self.q_std

    def mass_matrix(self, q):
        z = self.normalized_q(q)
        raw = self.mass_net(z)
        B = q.new_zeros((q.shape[0], 6, 6))
        vals = raw.clone()

        vals[:, self.diag_mask] = (F.softplus(vals[:, self.diag_mask]) + 1e-3)
        B[:, self.tri_i, self.tri_j] = vals

        eye = torch.eye(6, device=q.device, dtype=q.dtype)
        M = B @ B.transpose(-1, -2) + self.epsilon * eye
        return M

    def potential(self, q):
        return self.potential_net(self.normalized_q(q)).squeeze(-1)

    def damping(self):
        return F.softplus(self.raw_damping) + 1e-5 # tghats the damping offset

    def acceleration(self, q, qd, tau, create_graph=True):
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)

        M = self.mass_matrix(q)
        V = self.potential(q)

        gradV = torch.autograd.grad(
            V.sum(),
            q,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        Bsz = q.shape[0]
        dM_dq = q.new_zeros((Bsz, 6, 6, 6))

        for i in range(6):
            for j in range(i, 6):
                grad_entry = torch.autograd.grad(
                    M[:, i, j].sum(),
                    q,
                    create_graph=create_graph,
                    retain_graph=True,
                )[0]

                dM_dq[:, :, i, j] = grad_entry
                if i != j:
                    dM_dq[:, :, j, i] = grad_entry
        dotM = torch.einsum("bk,bkij->bij", qd, dM_dq)
        h = 0.5 * torch.einsum(
            "bj,bijl,bl->bi", qd, dM_dq, qd
        )

        Dqd = self.damping().unsqueeze(0) * qd

        dotM_qd = torch.bmm(
            dotM, qd.unsqueeze(-1)
        ).squeeze(-1)

        rhs = tau - Dqd - dotM_qd + h - gradV

        qdd = torch.linalg.solve(
            M, rhs.unsqueeze(-1)
        ).squeeze(-1)

        return qdd

    def forward(self, q, qd, tau, dt):
        qdd = self.acceleration(q, qd, tau)

        if not torch.is_tensor(dt):
            dt = q.new_tensor(dt)

        dt = dt.to(device=q.device, dtype=q.dtype).reshape(-1, 1)

        q_next = q + dt * qd + 0.5 * dt.square() * qdd
        qd_next = qd + dt * qdd

        return q_next, qd_next, qdd
