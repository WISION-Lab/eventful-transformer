from torch.linalg import vector_norm

from eventful_transformer.base import ExtendedModule


class TokenNormThreshold(ExtendedModule):
    """
    Defines a policy that selects tokens whose error norm exceeds a
    threshold.
    """
    def __init__(self, threshold=0.0, order=2):
        """
        :param threshold: The token norm threshold
        :param order: The type of norm (e.g., 2 for L2 norm)
        """
        super().__init__()
        self.threshold = threshold
        self.order = order

    def forward(self, x, dim=-1):
        """
        :param x: A tensor of token errors
        :param dim: The dimension along which we should reduce the norm
        """
        assert all(size == 1 for size in x.shape[:-2])

        # Note: The call to nonzero is very slow.
        index = vector_norm(x, ord=self.order, dim=dim).gt(self.threshold).nonzero()

        # Note: We assume here that all the leading dimensions (i.e.,
        # the batch dimension) have size 1. See the assertion above.
        return index[..., -1].view((1,) * (x.ndim - 2) + (-1,))

        # Alternative:
        # norm = vector_norm(x, ord=self.order, dim=-1)
        # return norm.topk(norm.gt(self.threshold).sum(), sorted=False)[1]


class TokenNormTopK(ExtendedModule):
    """
    Defines a policy that selects the k tokens with the largest error
    norm.
    """
    def __init__(self, k, order=2, save_status=False):
        """
        :param k: Select k tokens
        :param order: The type of norm (e.g., 2 for L2 norm)
        :param save_status: Cache inputs and outputs for debugging and
        visualization
        """
        super().__init__()
        self.k = k
        self.order = order
        self.save_status = save_status
        self.last_input = None
        self.last_output = None

    def forward(self, x, dim=-1):
        """
        :param x: A tensor of token errors
        :param dim: The dimension along which we should reduce the norm
        """
        output = vector_norm(x, ord=self.order, dim=dim).topk(self.k, sorted=False)[1]
        if self.save_status:
            # Clone to protect against external modification.
            self.last_input = x.clone()
            self.last_output = output.clone()
        return output


class TokenNormTopFraction(ExtendedModule):
    """
    Defines a policy that selects some fraction of tokens with the
    largest error norm.
    """
    def __init__(self, fraction, order=2):
        """
        :param fraction: Select this fraction of tokens (e.g., 0.5 for
        half of tokens)
        :param order: The type of norm (e.g., 2 for L2 norm)
        """
        super().__init__()
        assert not (fraction < 0.0 or fraction > 1.0)
        self.fraction = fraction
        self.order = order


    def forward(self, x, dim=-1):
        """
        :param x: A tensor of token errors
        :param dim: The dimension along which we should reduce the norm
        """
        x_norm = vector_norm(x, ord=self.order, dim=dim)
        k = int(self.fraction * x_norm.shape[-1])
        return x_norm.topk(k, sorted=False)[1]
