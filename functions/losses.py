import torch
from piq.base import BaseFeatureMetric
from piq.utils import _validate_input


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}



def _polynomial_kernel(X: torch.Tensor, Y: torch.Tensor = None, degree: int = 1, gamma = None,
                       coef0: float = 1.) -> torch.Tensor:
    """
    Compute the polynomial kernel between x and y
    K(X, Y) = (gamma <X, Y> + coef0)^degree
    Args:
        X: Tensor with shape (n_samples_1, n_features)
        Y: torch.Tensor of shape (n_samples_2, n_features)
        degree: default 3
        gamma: if None, defaults to 1.0 / n_features.
        coef0 : default 1
    Returns:
        Gram matrix : Array with shape (n_samples_1, n_samples_2)
    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html
    """

    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.size(1)

    K = torch.mm(X, Y.T)
    K *= gamma
    K += coef0
    K.pow_(degree)
    return K


def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor = None, gamma = None) -> torch.Tensor:
    """
    Compute the rbf kernel between X and Y
    K(X, Y) = exp(-1/2sigma ||X - Y||^2)
    Args:
        X: Tensor with shape (n_samples_1, n_features)
        Y: torch.Tensor of shape (n_samples_2, n_features)
        gamma: if None, defaults to 1.0 / n_features.
    Returns:
        Gram matrix : Array with shape (n_samples_1, n_samples_2)
    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html
    """

    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.size(1)

    X_norms = (X ** 2).sum(dim=1).view(-1, 1)
    Y_t = Y.t()
    Y_norms = (Y ** 2).sum(dim=1).view(1, -1)

    K = X_norms + Y_norms - 2.0 * torch.mm(X, Y_t)
    K *= -gamma
    K.exp_()
    return K




def _mmd2_and_variance(K_XX: torch.Tensor, K_XY: torch.Tensor, K_YY: torch.Tensor, unit_diagonal: bool = False,
                       mmd_est: str = 'unbiased', var_at_m = None, ret_var: bool = False): 
        
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.size(0)
    assert K_XX.size() == (m, m)
    assert K_XY.size() == (m, m)
    assert K_YY.size() == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
    else:
        diag_X = torch.diagonal(K_XX)
        diag_Y = torch.diagonal(K_YY)

       
    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)
 

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    # Compute the MMD^2 statistic.
    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1)) - 2 * K_XY_sum / (m * m)
  
    return mmd2


def _sqn(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.flatten()
    return flat.dot(flat)

def generate_random_projection(n_features: int, n_projections: int, device: torch.device = None) -> torch.Tensor:
    """
    Generates a random projection matrix of shape (n_features, n_projections)
    Args:
        n_features: number of features
        n_projections: number of projections
        device: device to store the matrix
    Returns:
        Tensor of shape (n_features, n_projections)
    """
    return torch.randn((n_features, n_projections), device=device)  
    
class KID(BaseFeatureMetric):
    r"""Interface of Kernel Inception Distance.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. KID can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.
    Args:
        degree: Degree of a polynomial functions used in kernels. Default: 3
        gamma: Kernel parameter. See paper for details
        coef0: Kernel parameter. See paper for details
        var_at_m: Kernel variance. Default is `None`
        average: If `True` recomputes metric `n_subsets` times using `subset_size` elements.
        n_subsets: Number of repeats. Ignored if `average` is False
        subset_size: Size of each subset for repeat. Ignored if `average` is False
        ret_var: Whether to return variance after the distance is computed.
            This function will return ``Tuple[torch.Tensor, torch.Tensor]`` in this case. Default: False
    Examples:
        >>> kid_metric = KID()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> kid: torch.Tensor = kid_metric(x_feats, y_feats)
    References:
        Demystifying MMD GANs https://arxiv.org/abs/1801.01401
    """

    def __init__(self, degree: int = 1, gamma = None, coef0: int = 1, var_at_m = None,
                 average: bool = False, n_subsets: int = 50, subset_size = 1000, ret_var: bool = False
                 ) -> None:
        super().__init__()

        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.ret_var = ret_var
        if average:
            self.n_subsets = n_subsets
            self.subset_size = subset_size
        else:
            self.n_subsets = 1
            self.subset_size = None

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor):
        """Computes KID (polynomial MMD) for given sets of features, obtained from Inception net
        or any other feature extractor.
        Samples must be in range [0, 1].
        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`
        Returns:
            KID score and variance (optional).
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(1, 2))
        var_at_m = min(x_features.size(0), y_features.size(0))
        if self.subset_size is None:
            subset_size = x_features.size(0)
        else:
            subset_size = self.subset_size

        results = []
        for _ in range(self.n_subsets):
            x_subset = x_features[torch.randperm(len(x_features))[:subset_size]]
            y_subset = y_features[torch.randperm(len(y_features))[:subset_size]]

            # use  k(x, y) = (gamma <x, y> + coef0)^degree
            # default gamma is 1 / dim
            K_XX = _polynomial_kernel(
                x_subset,
                None,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0)
            K_YY = _polynomial_kernel(
                y_subset,
                None,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0)
            K_XY = _polynomial_kernel(
                x_subset,
                y_subset,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0)

            out = _mmd2_and_variance(K_XX, K_XY, K_YY, var_at_m=var_at_m, ret_var=self.ret_var)
            results.append(out)

       
        score = torch.mean(torch.stack(results, dim=0))
        return score


class KID_rbf(BaseFeatureMetric):
    r"""Interface of Kernel Inception Distance.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. KID can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.
    Args:
        degree: Degree of a polynomial functions used in kernels. Default: 3
        gamma: Kernel parameter. See paper for details
        coef0: Kernel parameter. See paper for details
        var_at_m: Kernel variance. Default is `None`
        average: If `True` recomputes metric `n_subsets` times using `subset_size` elements.
        n_subsets: Number of repeats. Ignored if `average` is False
        subset_size: Size of each subset for repeat. Ignored if `average` is False
        ret_var: Whether to return variance after the distance is computed.
            This function will return ``Tuple[torch.Tensor, torch.Tensor]`` in this case. Default: False
    Examples:
        >>> kid_metric = KID()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> kid: torch.Tensor = kid_metric(x_feats, y_feats)
    References:
        Demystifying MMD GANs https://arxiv.org/abs/1801.01401
    """

    def __init__(self, degree: int = 1, gamma = None, coef0: int = 1, var_at_m = None,
                 average: bool = False, n_subsets: int = 50, subset_size = 1000, ret_var: bool = False
                 ) -> None:
        super().__init__()

        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.ret_var = ret_var
        if average:
            self.n_subsets = n_subsets
            self.subset_size = subset_size
        else:
            self.n_subsets = 1
            self.subset_size = None


    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor):
        """Computes KID (polynomial MMD) for given sets of features, obtained from Inception net
        or any other feature extractor.
        Samples must be in range [0, 1].
        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`
        Returns:
            KID score and variance (optional).
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(1, 2))
        var_at_m = min(x_features.size(0), y_features.size(0))
        if self.subset_size is None:
            subset_size = x_features.size(0)
        else:
            subset_size = self.subset_size

        results = []
        for _ in range(self.n_subsets):
            x_subset = x_features[torch.randperm(len(x_features))[:subset_size]]
            y_subset = y_features[torch.randperm(len(y_features))[:subset_size]]

            K_XX = _rbf_kernel(
                x_subset,
                None,
                gamma=self.gamma)

            K_YY = _rbf_kernel(
                y_subset,
                None,
                gamma=self.gamma)

            K_XY = _rbf_kernel(
                x_subset,
                y_subset,
                gamma=self.gamma)

            out = _mmd2_and_variance(K_XX, K_XY, K_YY, var_at_m=var_at_m, ret_var=self.ret_var)
            results.append(out)

       
        score = torch.mean(torch.stack(results, dim=0))
        return score


