import torch
import torch.nn as nn

def neg_partial_log_likelihood(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Cox negative partial log-likelihood (to minimize).

    Args:
        risk_scores: Tensor of shape [N], higher means higher risk
        times: Tensor of shape [N]
        events: Tensor of shape [N], 1=event, 0=censored
    """
    if risk_scores.dim() > 1:
        risk_scores = risk_scores.view(-1)
    sort_idx = torch.argsort(times, descending=True)
    pred_sorted = risk_scores[sort_idx].view(-1, 1)
    event_sorted = events[sort_idx].view(-1, 1).float()

    n = pred_sorted.size(0)
    if n == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
    if event_sorted.sum() == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
    
    # If only one sample, return a small regularization loss
    if n == 1:
        return torch.tensor(0.1, device=risk_scores.device, requires_grad=True)

    indicator = torch.tril(torch.ones((n, n), device=pred_sorted.device, dtype=pred_sorted.dtype))
    risk_set_sum = indicator @ torch.exp(pred_sorted)
    risk_set_sum = torch.clamp(risk_set_sum, min=1e-12)

    log_likelihood = pred_sorted - torch.log(risk_set_sum)
    num_events = torch.clamp(event_sorted.sum(), min=1.0)
    loss = -(log_likelihood * event_sorted).sum() / num_events
    
    # Add small regularization to prevent zero loss
    loss = loss + 1e-6 * torch.norm(risk_scores)
    return loss


def compute_c_index(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> float:
    """Compute Harrell's C-index.

    Returns float in [0,1]. If insufficient comparable pairs, returns NaN.
    """
    with torch.no_grad():
        r = risk_scores.detach().cpu().view(-1).numpy()
        t = times.detach().cpu().view(-1).numpy()
        e = events.detach().cpu().view(-1).numpy()

        n = len(r)
        num = 0
        den = 0
        for i in range(n):
            for j in range(n):
                if t[i] < t[j] and e[i] == 1:
                    den += 1
                    if r[i] > r[j]:
                        num += 1
                    elif r[i] == r[j]:
                        num += 0.5
        if den == 0:
            return float('nan')
        return num / den


class CoxLoss(nn.Module):
    def forward(self, risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        return neg_partial_log_likelihood(risk_scores, times, events)


