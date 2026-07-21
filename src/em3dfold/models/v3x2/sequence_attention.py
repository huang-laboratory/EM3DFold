import math
from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from em3dfold.utils.torch_utils import get_batch_slices, padded_sequence_softmax


def get_batched_sequence_attention_scores(
    sequence_query,
    sequence_key,
    batch,
    attention_scale,
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_query.shape[0],
        sequence_key.shape[1],
        sequence_key.shape[2],
        device=device,
    )

    seq_batches = get_batch_slices(output.shape[1], batch_size)
    sequence_query = sequence_query[:, None]

    for seq_batch in seq_batches:
        output[:, seq_batch] = (
            sequence_query * sequence_key[:, seq_batch][batch]
        ).sum(dim=-1) / attention_scale

    return output


def get_batched_sequence_attention_features(
    sequence_attention_weights,
    sequence_value,
    batch,
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_attention_weights.shape[0],
        sequence_value.shape[2],
        sequence_value.shape[3],
        device=device,
    )
    seq_batches = get_batch_slices(sequence_attention_weights.shape[1], batch_size)

    for seq_batch in seq_batches:
        output += (
            sequence_attention_weights[:, seq_batch][..., None]
            * sequence_value[:, seq_batch][batch]
        ).sum(dim=1)
    return output


class SequenceAttention(nn.Module):
    def __init__(
        self,
        d: int = 256,
        d_seq: int = 1280,
        d_seq_na: int | None = None,
        d_head: int = 48,
        n_head: int = 8,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.d = d
        self.d_seq_prot = d_seq
        self.d_seq_na = d_seq if d_seq_na is None else d_seq_na
        self.d_seq_joint = self.d_seq_prot + self.d_seq_na
        self.d_head = d_head
        self.n_head = n_head

        self.attention_scale = math.sqrt(self.d_head)
        self.norm = nn.LayerNorm(self.d)

        self.q_prot = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.q_na = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )

        self.k_prot = nn.Sequential(
            nn.Linear(self.d_seq_prot, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.v_prot = nn.Sequential(
            nn.Linear(self.d_seq_prot, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.k_na = nn.Sequential(
            nn.Linear(self.d_seq_na, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.v_na = nn.Sequential(
            nn.Linear(self.d_seq_na, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )

        self.gate_prot = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
            nn.Sigmoid(),
        )
        self.gate_na = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
            nn.Sigmoid(),
        )

        self.back_prot = nn.Sequential(
            Rearrange("... h d -> ... (h d)", h=self.n_head, d=self.d_head),
            nn.Linear(self.n_head * self.d_head, self.d, bias=False),
        )
        self.back_na = nn.Sequential(
            Rearrange("... h d -> ... (h d)", h=self.n_head, d=self.d_head),
            nn.Linear(self.n_head * self.d_head, self.d, bias=False),
        )
        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def _prepare_sequence_inputs(
        self,
        seq_emb,
        seq_mask,
        d_seq,
        batch_size,
        device,
        dtype,
    ):
        if seq_emb is None:
            seq_emb = torch.zeros(batch_size, 3, d_seq, device=device, dtype=dtype)
            seq_mask = torch.ones(batch_size, 3, device=device, dtype=torch.bool)
            return seq_emb, seq_mask

        if seq_mask is None:
            seq_mask = torch.ones(seq_emb.shape[:2], device=seq_emb.device, dtype=torch.bool)

        return seq_emb, seq_mask

    def _attend_subset(
        self,
        node_subset,
        batch_subset,
        sequence_emb,
        sequence_mask,
        attention_batch_size,
        device,
        q_module,
        k_module,
        v_module,
        gate_module,
        back_module,
    ):
        y = self.norm(node_subset)
        sequence_query = q_module(y)
        sequence_key = k_module(sequence_emb)
        sequence_value = v_module(sequence_emb)

        sequence_attention_scores = get_batched_sequence_attention_scores(
            sequence_query,
            sequence_key,
            batch_subset,
            self.attention_scale,
            batch_size=attention_batch_size,
            device=device,
        )

        batched_mask = sequence_mask[batch_subset].unsqueeze(-1)
        sequence_attention_weights = padded_sequence_softmax(
            sequence_attention_scores, batched_mask, dim=1
        )

        new_features_attention = get_batched_sequence_attention_features(
            sequence_attention_weights,
            sequence_value,
            batch_subset,
            batch_size=attention_batch_size,
            device=device,
        )

        gate = gate_module(y)
        new_features = back_module(gate * new_features_attention)
        return new_features

    def forward_normal(
        self,
        node,
        prot_mask=None,
        batch=None,
        attention_batch_size=200,
        prot_seq_emb=None,
        prot_seq_mask=None,
        na_seq_emb=None,
        na_seq_mask=None,
        **kwargs,
    ):
        return self._intern_forward(
            node=node,
            prot_mask=prot_mask,
            batch=batch,
            attention_batch_size=attention_batch_size,
            prot_seq_emb=prot_seq_emb,
            prot_seq_mask=prot_seq_mask,
            na_seq_emb=na_seq_emb,
            na_seq_mask=na_seq_mask,
        )

    def forward_checkpoint(
        self,
        node,
        prot_mask=None,
        batch=None,
        attention_batch_size=200,
        prot_seq_emb=None,
        prot_seq_mask=None,
        na_seq_emb=None,
        na_seq_mask=None,
        **kwargs,
    ):
        if (not torch.is_grad_enabled()) or (not node.requires_grad):
            return self._intern_forward(
                node=node,
                prot_mask=prot_mask,
                batch=batch,
                attention_batch_size=attention_batch_size,
                prot_seq_emb=prot_seq_emb,
                prot_seq_mask=prot_seq_mask,
                na_seq_emb=na_seq_emb,
                na_seq_mask=na_seq_mask,
            )
        new_forward = partial(
            self._intern_forward,
            prot_mask=prot_mask,
            batch=batch,
            attention_batch_size=attention_batch_size,
            prot_seq_emb=prot_seq_emb,
            prot_seq_mask=prot_seq_mask,
            na_seq_emb=na_seq_emb,
            na_seq_mask=na_seq_mask,
        )
        return torch.utils.checkpoint.checkpoint(
            new_forward,
            node,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def _intern_forward(
        self,
        node,
        prot_mask,
        batch,
        attention_batch_size,
        prot_seq_emb,
        prot_seq_mask,
        na_seq_emb,
        na_seq_mask,
    ):
        device = node.device
        if prot_mask is None:
            raise ValueError("prot_mask must be provided for mixed protein/NA sequence attention.")

        if batch is None:
            batch = torch.zeros(node.shape[0], dtype=torch.long, device=device)

        prot_mask = prot_mask.bool()
        na_mask = ~prot_mask
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        prot_seq_emb, prot_seq_mask = self._prepare_sequence_inputs(
            prot_seq_emb,
            prot_seq_mask,
            self.d_seq_prot,
            batch_size,
            device,
            node.dtype,
        )
        na_seq_emb, na_seq_mask = self._prepare_sequence_inputs(
            na_seq_emb,
            na_seq_mask,
            self.d_seq_na,
            batch_size,
            device,
            node.dtype,
        )

        new_features_all = torch.zeros_like(node)

        prot_features = self._attend_subset(
            node_subset=node[prot_mask],
            batch_subset=batch[prot_mask],
            sequence_emb=prot_seq_emb,
            sequence_mask=prot_seq_mask,
            attention_batch_size=attention_batch_size,
            device=device,
            q_module=self.q_prot,
            k_module=self.k_prot,
            v_module=self.v_prot,
            gate_module=self.gate_prot,
            back_module=self.back_prot,
        )
        prot_features = prot_features.to(dtype=new_features_all.dtype)
        new_features_all[prot_mask] = prot_features

        na_features = self._attend_subset(
            node_subset=node[na_mask],
            batch_subset=batch[na_mask],
            sequence_emb=na_seq_emb,
            sequence_mask=na_seq_mask,
            attention_batch_size=attention_batch_size,
            device=device,
            q_module=self.q_na,
            k_module=self.k_na,
            v_module=self.v_na,
            gate_module=self.gate_na,
            back_module=self.back_na,
        )
        na_features = na_features.to(dtype=new_features_all.dtype)
        new_features_all[na_mask] = na_features

        return math.sqrt(2) * node + new_features_all


def _grad_status(module):
    status = {}
    for name, param in module.named_parameters():
        if param.grad is None:
            status[name] = "none"
        else:
            grad_norm = float(param.grad.norm().item())
            status[name] = f"{grad_norm:.6e}"
    return status


def _run_grad_debug_case(
    name,
    prot_mask,
    provide_prot_seq=True,
    provide_na_seq=True,
    checkpoint=False,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SequenceAttention(
        checkpoint=checkpoint,
        d=256,
        d_seq=1280,
        d_seq_na=640,
        d_head=32,
        n_head=8,
    ).to(device)
    model.train()

    n = prot_mask.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    node = torch.randn(n, 256, device=device, requires_grad=True)
    prot_mask = prot_mask.to(device=device, dtype=torch.bool)

    prot_seq_emb = None
    prot_seq_mask = None
    na_seq_emb = None
    na_seq_mask = None

    if provide_prot_seq:
        prot_seq_emb = torch.randn(1, 32, 1280, device=device)
        prot_seq_mask = torch.ones(1, 32, dtype=torch.bool, device=device)
    if provide_na_seq:
        na_seq_emb = torch.randn(1, 24, 640, device=device)
        na_seq_mask = torch.ones(1, 24, dtype=torch.bool, device=device)

    out = model(
        node=node,
        prot_mask=prot_mask,
        batch=batch,
        prot_seq_emb=prot_seq_emb,
        prot_seq_mask=prot_seq_mask,
        na_seq_emb=na_seq_emb,
        na_seq_mask=na_seq_mask,
    )
    loss = out.square().mean()
    loss.backward()

    grad_map = _grad_status(model)
    print(f"# case: {name}")
    print(f"# loss: {float(loss.item()):.6f}")
    for key in [
        "q_prot.0.weight",
        "k_prot.0.weight",
        "v_prot.0.weight",
        "gate_prot.0.weight",
        "back_prot.1.weight",
        "q_na.0.weight",
        "k_na.0.weight",
        "v_na.0.weight",
        "gate_na.0.weight",
        "back_na.1.weight",
    ]:
        print(f"# grad {key}: {grad_map[key]}")
    print("#")


if __name__ == "__main__":
    _run_grad_debug_case(
        name="mixed_nodes_with_both_sequences",
        prot_mask=torch.tensor([True, True, False, False, True, False]),
        provide_prot_seq=True,
        provide_na_seq=True,
        checkpoint=False,
    )
    _run_grad_debug_case(
        name="protein_only_nodes_with_both_sequences",
        prot_mask=torch.tensor([True, True, True, True, True, True]),
        provide_prot_seq=True,
        provide_na_seq=True,
        checkpoint=False,
    )
    _run_grad_debug_case(
        name="na_only_nodes_with_both_sequences",
        prot_mask=torch.tensor([False, False, False, False, False, False]),
        provide_prot_seq=True,
        provide_na_seq=True,
        checkpoint=False,
    )
    _run_grad_debug_case(
        name="protein_only_nodes_without_na_sequence",
        prot_mask=torch.tensor([True, True, True, True, True, True]),
        provide_prot_seq=True,
        provide_na_seq=False,
        checkpoint=False,
    )
    _run_grad_debug_case(
        name="na_only_nodes_without_protein_sequence",
        prot_mask=torch.tensor([False, False, False, False, False, False]),
        provide_prot_seq=False,
        provide_na_seq=True,
        checkpoint=False,
    )
