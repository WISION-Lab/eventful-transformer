from eventful_transformer.base import ExtendedModule
from eventful_transformer.counting import CountedMatmul
from eventful_transformer.utils import expand_col_index, expand_row_index


class SimpleSTGTGate(ExtendedModule):
    """
    This class implements a simple version of the gating logic described
    in "Spatio-Temporal Gated Transformers for Efficient Video
    Processing". This is intended to be used as an experimental
    baseline.
    """

    def __init__(self, structure="row"):
        """
        :param structure: Options other than structure="row" have not
        yet been implemented
        """
        super().__init__()

        # Currently,
        assert structure == "row"

        self.first = True
        self.policy = None
        self.p = None

    def forward(self, c):
        if self.first:
            return self.forward_first(c)
        else:
            return self.forward_incremental(c)

    def forward_first(self, c):
        self.first = False
        self.p = c
        return c, None

    def forward_incremental(self, c):
        if self.count_mode:
            self.counts["gate_flops"] += c.numel()
        index = self.policy(c - self.p, dim=-1)
        c_tilde = c.gather(dim=-2, index=expand_row_index(index, c.shape))
        self.p = c
        return c_tilde, index

    def reset_self(self):
        self.first = True
        self.p = None


class TokenBuffer(ExtendedModule):
    """
    Defines a token buffer.
    """

    def __init__(self, structure="row"):
        """
        :param structure: Whether tokens should be indexed along the
        last ("col") or second-to-last ("row") dimension
        """
        super().__init__()
        assert structure in ["row", "col"]
        self.structure = structure
        self.first = True
        self.b = None

    def forward(self, x, index):
        """
        Warning - the output is a direct reference to self.b (a state
        tensor).
        """
        if self.first:
            return self.forward_first(x)
        else:
            return self.forward_incremental(x, index)

    def forward_first(self, x):
        """
        Forward pass on the first time step (flush).
        """
        self.first = False
        self.b = x.clone()
        return self.b

    def forward_incremental(self, x, index):
        """
        Forward pass after the first time step (incremental update).
        """
        if self.structure == "row":
            index = expand_row_index(index, self.b.shape)
            dim = -2
        else:
            index = expand_col_index(index, self.b.shape)
            dim = -1
        self.b.scatter_(dim=dim, index=index, src=x)
        return self.b

    def reset_self(self):
        self.first = True
        self.b = None


class TokenGate(ExtendedModule):
    """
    Defines a token gate.

    TokenGate.policy defines the token selection policy.
    """

    def __init__(self, structure="row"):
        """
        :param structure: Whether tokens should be indexed along the
        last ("col") or second-to-last ("row") dimension
        """
        super().__init__()
        assert structure in ["row", "col"]
        self.structure = structure
        self.first = True
        self.policy = None
        self.p = None

    def forward(self, c, forced_index=None):
        """
        :param c: Warning - self.p (a state tensor) retains a direct
        reference to the last value of this input
        :param forced_index: A set of indices to force-update (instead
        of letting the policy decide)
        """
        if self.first:
            return self.forward_first(c)
        else:
            return self.forward_incremental(c, forced_index=forced_index)

    def forward_first(self, c):
        """
        Forward pass on the first time step (flush).
        """
        self.first = False
        self.p = c
        return c, None

    def forward_incremental(self, c, forced_index=None):
        """
        Forward pass after the first time step (incremental update).
        """
        if self.count_mode:
            self.counts["gate_flops"] += self.p.numel()
        dim, expanded, index = self._apply_policy(c - self.p, forced_index)
        c_tilde = c.gather(dim=dim, index=expanded)
        self.p.scatter_(dim=dim, index=expanded, src=c_tilde)
        return c_tilde, index

    def _apply_policy(self, x, forced_index):
        dim = -2 if (self.structure == "row") else -1
        if forced_index is None:
            index = self.policy(x, dim=(-1 if (self.structure == "row") else -2))
        else:
            index = forced_index
        if self.structure == "row":
            expanded = expand_row_index(index, x.shape)
        else:
            expanded = expand_col_index(index, x.shape)
        return dim, expanded, index

    def reset_self(self):
        self.first = True
        self.p = None


class TokenDeltaGate(TokenGate):
    """
    Defines a token delta gate.
    """

    def __init__(self, structure="row"):
        """
        :param structure: Whether tokens should be indexed along the
        last ("col") or second-to-last ("row") dimension
        """
        super().__init__(structure=structure)

    def forward_first(self, c):
        c = super().forward_first(c)[0]
        return c, None, None

    def forward_incremental(self, c, forced_index=None):
        """
        :param c: Warning - self.p (a state tensor) retains a direct
        reference to the last value of this input
        :param forced_index: A set of indices to force-update (instead
        of letting the policy decide)
        """
        if self.count_mode:
            self.counts["gate_flops"] += self.p.numel()
        e = c - self.p
        dim, expanded, index = self._apply_policy(e, forced_index)
        c_tilde = c.gather(dim=dim, index=expanded)
        e_tilde = e.gather(dim=dim, index=expanded)
        self.p.scatter_(dim=dim, index=expanded, src=c_tilde)
        return c_tilde, e_tilde, index


class MatmulBuffer(ExtendedModule):
    """
    Defines a buffer for updating the query-key product.
    """
    def __init__(self):
        super().__init__()
        self.first = True
        self.product = None
        self.matmul = CountedMatmul()

    def forward(self, q, k, index_q, index_k):
        """
        Warning - the output is a direct reference to self.product (a
        state tensor).
        """
        if self.first:
            return self.forward_first(q, k)
        else:
            return self.forward_incremental(q, k, index_q, index_k)

    def forward_first(self, q, k):
        """
        Forward pass on the first time step (flush).
        """
        self.first = False
        self.product = self.matmul(q, k)
        return self.product

    def forward_incremental(self, q, k, index_q, index_k):
        """
        Forward pass after the first time step (incremental update).
        """
        q_tilde = q.gather(dim=-2, index=expand_row_index(index_q, q.shape))
        k_tilde = k.gather(dim=-1, index=expand_col_index(index_k, k.shape))
        self.product.scatter_(
            dim=-2,
            index=expand_row_index(index_q, self.product.shape),
            src=self.matmul(q_tilde, k),
        )
        self.product.scatter_(
            dim=-1,
            index=expand_col_index(index_k, self.product.shape),
            src=self.matmul(q, k_tilde),
        )
        return self.product

    def reset_self(self):
        self.first = True
        self.product = None


class MatmulDeltaAccumulator(ExtendedModule):
    """
    Defines a buffer for updating the attention-value product.
    """
    def __init__(self):
        super().__init__()
        self.first = True
        self.product = None
        self.matmul = CountedMatmul()

    def forward(self, a_n_tilde, v_n_tilde, a_delta_tilde, v_delta_tilde):
        """
        Warning - the output is a direct reference to self.product (a
        state tensor).
        """
        if self.first:
            return self.forward_first(a_n_tilde, v_n_tilde)
        else:
            return self.forward_incremental(
                a_n_tilde, v_n_tilde, a_delta_tilde, v_delta_tilde
            )

    def forward_first(self, a, v):
        """
        Forward pass on the first time step (flush).
        """
        self.first = False
        self.product = self.matmul(a, v)
        return self.product

    def forward_incremental(self, a_n_tilde, v_n_tilde, a_delta_tilde, v_delta_tilde):
        """
        Forward pass after the first time step (incremental update).
        """
        if self.count_mode:
            self.counts["accumulator_flops"] += (
                v_n_tilde.numel() + 2 * self.product.numel()
            )
        self.product += self.matmul(a_n_tilde, v_delta_tilde)
        self.product += self.matmul(a_delta_tilde, v_n_tilde - v_delta_tilde)
        return self.product

    def reset_self(self):
        self.first = True
        self.product = None
