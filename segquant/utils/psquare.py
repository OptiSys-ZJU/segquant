import torch


class PSquare:
    """
    Batch P² algorithm for dynamic quantile estimation.
    Estimates the p-quantile per channel without storing all data.
    """

    def __init__(self, k, p=50, device="cpu"):
        """
        :param k: number of channels (batch dimension)
        :param p: percentile to track (0-100)
        :param device: torch device
        """
        self.k = k
        self.p = p / 100.0
        self.device = device
        self.initiated = False

        # marker heights and positions
        self.marker_heights = torch.empty((k, 0), device=device, dtype=torch.float32)
        self.marker_positions = (
            torch.arange(1, 6, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(k, 1)
        )
        self.desired_positions = (
            torch.tensor(
                [1.0, 1.0 + 2 * self.p, 1.0 + 4 * self.p, 3.0 + 2 * self.p, 5.0],
                device=device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(k, 1)
        )
        self.increments = (
            torch.tensor(
                [0.0, self.p / 2, self.p, (1 + self.p) / 2, 1.0],
                device=device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(k, 1)
        )

    def find_cell(self, obs: torch.Tensor):
        """
        For each channel, find the highest marker <= obs
        :param obs: tensor(k,)
        :return: tensor(k,) with index -1..4
        """
        ge = obs[:, None] >= self.marker_heights  # (k,5)
        i = ge.sum(dim=1) - 1
        i[i < 0] = -1
        return i

    def _parabolic(self, i, d, rows):
        """
        Batched parabolic update for rows
        :param i: int, marker index
        :param d: tensor(num_rows,)
        :param rows: tensor(num_rows,)
        """
        h = self.marker_heights[rows]
        pos = self.marker_positions[rows]

        term1 = d / (pos[:, i + 1] - pos[:, i - 1])
        term2 = (
            (pos[:, i] - pos[:, i - 1] + d)
            * (h[:, i + 1] - h[:, i])
            / (pos[:, i + 1] - pos[:, i])
        )
        term3 = (
            (pos[:, i + 1] - pos[:, i] - d)
            * (h[:, i] - h[:, i - 1])
            / (pos[:, i] - pos[:, i - 1])
        )
        return h[:, i] + term1 * (term2 + term3)

    def _linear(self, i, d, rows):
        """
        Batched linear update for rows
        """
        h = self.marker_heights[rows]
        pos = self.marker_positions[rows]
        idx_to = (i + d).long()
        return h[:, i] + d * (h.gather(1, idx_to[:, None])[:, 0] - h[:, i]) / (
            pos.gather(1, idx_to[:, None])[:, 0] - pos[:, i]
        )

    def update(self, obs: torch.Tensor):
        """
        Update P² markers for each channel
        :param obs: tensor(k,)
        """
        if not self.initiated:
            self.marker_heights = torch.cat([self.marker_heights, obs[:, None]], dim=1)
            if self.marker_heights.shape[1] == 5:
                self.initiated = True
                self.marker_heights, _ = torch.sort(self.marker_heights, dim=1)
            return

        i = self.find_cell(obs)
        mask_lower = i == -1
        mask_upper = i == 4
        self.marker_heights[mask_lower, 0] = obs[mask_lower]
        self.marker_heights[mask_upper, 4] = obs[mask_upper]

        k_vals = i.clone()
        k_vals[mask_lower] = 0
        k_vals[mask_upper] = 3

        for ch in range(self.k):
            k_ch = k_vals[ch]
            if k_ch < 4:
                self.marker_positions[ch, k_ch + 1 :] += 1

        self.desired_positions += self.increments
        self.adjust_height_values()

    def adjust_height_values(self):
        """
        Step B.3: adjust marker heights using parabolic or linear formula
        """
        for i in range(1, 4):
            d = self.desired_positions[:, i] - self.marker_positions[:, i]
            mask = (
                (d >= 1)
                & ((self.marker_positions[:, i + 1] - self.marker_positions[:, i]) > 1)
            ) | (
                (d <= -1)
                & ((self.marker_positions[:, i - 1] - self.marker_positions[:, i]) < -1)
            )

            if mask.any():
                rows = torch.nonzero(mask, as_tuple=True)[0]
                d_rows = d[rows].sign()  # -1 or 1

                q_par = self._parabolic(i, d_rows, rows)
                q_lin = self._linear(i, d_rows, rows)

                valid = (self.marker_heights[rows, i - 1] < q_par) & (
                    q_par < self.marker_heights[rows, i + 1]
                )
                self.marker_heights[rows, i] = torch.where(valid, q_par, q_lin)
                self.marker_positions[rows, i] += d_rows

    def p_estimate(self):
        """
        Return current estimate of the p-quantile for each channel
        """
        if self.initiated:
            return self.marker_heights[:, 2]
        else:
            return torch.full((self.k,), float("nan"), device=self.device)
