import torch


class GaussNoise(object):
    def __init__(
        self,
        sigma: float = 0.05,
    ):
        assert isinstance(sigma, (int, float))
        self.sigma = sigma

    def __call__(self, img: torch.Tensor):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)

        out = img + self.sigma * torch.randn_like(img)

        if out.dtype != dtype:
            out = out.to(dtype)

        return out

    def __repr__(self):
        return self.__class__.__name__ + f'(sigma={self.sigma})'
