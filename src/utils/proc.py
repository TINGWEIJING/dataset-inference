from typing import List
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


class MinMaxScaler(object):
    def __init__(
        self,
        channel_min: List[float],
        channel_max: List[float],
        new_channel_min: List[float],
        new_channel_max: List[float],
    ):
        assert isinstance(channel_min, list)
        assert isinstance(channel_max, list)
        assert isinstance(new_channel_min, list)
        assert isinstance(new_channel_max, list)
        self.channel_min = channel_min
        self.channel_max = channel_max
        self.new_channel_min = new_channel_min
        self.new_channel_max = new_channel_max

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Only support RGB image (3 channels)
        '''
        assert isinstance(tensor, torch.Tensor)
        dtype = tensor.dtype

        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if tensor.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                             '{}.'.format(tensor.size()))

        src_min = torch.as_tensor(self.channel_min, dtype=dtype, device=tensor.device)
        src_max = torch.as_tensor(self.channel_max, dtype=dtype, device=tensor.device)
        new_min = torch.as_tensor(self.new_channel_min, dtype=dtype, device=tensor.device)
        new_max = torch.as_tensor(self.new_channel_max, dtype=dtype, device=tensor.device)

        # for batch images
        if tensor.ndim == 4:
            src_min = src_min.view(1, 3, 1, 1).expand(tensor.shape[0], -1, -1, -1)
            src_max = src_max.view(1, 3, 1, 1).expand(tensor.shape[0], -1, -1, -1)
            new_min = new_min.view(1, 3, 1, 1).expand(tensor.shape[0], -1, -1, -1)
            new_max = new_max.view(1, 3, 1, 1).expand(tensor.shape[0], -1, -1, -1)
        else:
            src_min = src_min.view(3, 1, 1)
            src_max = src_max.view(3, 1, 1)
            new_min = new_min.view(3, 1, 1)
            new_max = new_max.view(3, 1, 1)

        return (tensor - src_min)/(src_max - src_min)*(new_max - new_min) + new_min

    def __repr__(self):
        return self.__class__.__name__ + f'(channel_min={self.channel_min}, channel_max={self.channel_max}, new_channel_min={self.new_channel_min}, new_channel_max={self.new_channel_max})'
