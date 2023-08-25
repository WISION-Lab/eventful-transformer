import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as func
from torchvision import transforms


def as_float32(x):
    if isinstance(x, torch.Tensor) and x.dtype == torch.uint8:
        return x.float() / 255.0
    elif isinstance(x, np.ndarray) and x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    elif type(x) in (tuple, list) and isinstance(x[0], int):
        return type(x)(x_i / 255.0 for x_i in x)
    else:
        return x


def as_uint8(x):
    if isinstance(x, torch.Tensor) and x.dtype != torch.uint8:
        return (x * 255.0).byte()
    elif isinstance(x, np.ndarray) and x.dtype != np.uint8:
        return (x * 255.0).astype(np.uint8)
    elif type(x) in (tuple, list) and isinstance(x[0], float):
        return type(x)(int(x_i * 255.0) for x_i in x)
    else:
        return x


def pad_to_size(x, size, pad_tensor=None):
    # padding = [0, size[1] - x.shape[-1], 0, size[0] - x.shape[-2]]
    # x = func.pad(x, padding, fill=0, padding_mode="constant")
    # The two lines above are not working as expected - maybe there's a
    # bug in func.pad? In the meantime we'll use the concat-based
    # padding code below.
    if pad_tensor is None:
        pad_tensor = torch.zeros((1,) * x.ndim, dtype=x.dtype, device=x.device)
    for dim in list(range(-1, -len(size) - 1, -1)):
        expand_shape = list(x.shape)
        expand_shape[dim] = size[dim] - x.shape[dim]
        if expand_shape[dim] == 0:
            continue

        # torch.concat allocates a new tensor. So, we're safe to use
        # torch.expand here (instead of torch.repeat) without worrying
        # about different elements of x referencing the same data.
        x = torch.concat([x, pad_tensor.expand(expand_shape)], dim)
    return x


def rescale(
    x, scale, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
):
    if scale != 1.0:
        x = func.resize(
            x,
            [round(scale * x.shape[-2]), round(scale * x.shape[-1])],
            interpolation=interpolation,
            antialias=antialias,
        )
    return x


def resize_to_fit(
    x, size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
):
    scale = max(size[0] / x.shape[-2], size[1] / x.shape[-1])
    x = rescale(x, scale, interpolation=interpolation, antialias=antialias)
    return x


def save_image_mpl(image, pathname, **imshow_kwargs):
    fig, ax = plt.subplots()
    ax.imshow(image, **imshow_kwargs)
    ax.axis("off")
    fig.savefig(pathname, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def write_image(filename, image, **kwargs):
    filename = str(filename)
    lower = filename.lower()
    image = torch.as_tensor(image)
    assert any(lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])
    if lower.endswith(".png"):
        torchvision.io.write_png(image, filename, **kwargs)
    else:
        torchvision.io.write_jpeg(image, filename, **kwargs)


def write_video(filename, video, fps=30, is_chw=True):
    filename = str(filename)
    video = torch.as_tensor(video)
    if is_chw:
        video = video.permute(0, 2, 3, 1)
    torchvision.io.write_video(filename, video, fps=fps)
