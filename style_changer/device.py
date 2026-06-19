import os


def select_device() -> str:
    """Pick a torch device: STYLE_DEVICE override, else cuda -> mps -> cpu."""
    override = os.environ.get("STYLE_DEVICE")
    if override:
        return override

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
