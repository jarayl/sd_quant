from typing import Any, List, Optional, Sequence, Union
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics import Metric 
# from pytorch_lightning.metrics import Metric
# from torchmetrics.functional.multimodal.clip_score import _clip_score_update, _get_clip_model_and_processor

# from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
# from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_10
# from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

# if not _MATPLOTLIB_AVAILABLE:
#     __doctest_skip__ = ["CLIPScore.plot"]

# if _SKIP_SLOW_DOCTEST and _TRANSFORMERS_GREATER_EQUAL_4_10:
from transformers import CLIPModel as _CLIPModel
from transformers import CLIPProcessor as _CLIPProcessor

# def _download_clip_for_clip_score() -> None:
#     _CLIPModel.from_pretrained("openai/clip-vit-large-patch14", resume_download=True)
#     _CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", resume_download=True)

# if not _try_proceed_with_timeout(_download_clip_for_clip_score):
#     __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]
# else:
#     __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]


# HELPER FUNCTIONS

def _clip_score_update(
    images: Union[Tensor, List[Tensor]],
    text: Union[str, List[str]],
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> Tuple[Tensor, int]:
    if not isinstance(images, list):
        if images.ndim == 3:
            images = [images]
    else:  # unwrap into list
        images = list(images)

    if not all(i.ndim == 3 for i in images):
        raise ValueError("Expected all images to be 3d but found image that has either more or less")

    if not isinstance(text, list):
        text = [text]

    if len(text) != len(images):
        raise ValueError(
            f"Expected the number of images and text examples to be the same but got {len(images)} and {len(text)}"
        )
    device = images[0].device
    processed_input = processor(text=text, images=[i.cpu() for i in images], return_tensors="pt", padding=True)

    img_features = model.get_image_features(processed_input["pixel_values"].to(device))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    max_position_embeddings = model.config.text_config.max_position_embeddings
    # if processed_input["attention_mask"].shape[-1] > max_position_embeddings:
    #     rank_zero_warn(
    #         f"Encountered caption longer than {max_position_embeddings=}. Will truncate captions to this length."
    #         "If longer captions are needed, initialize argument `model_name_or_path` with a model that supports"
    #         "longer sequences",
    #         UserWarning,
    #     )
    #     processed_input["attention_mask"] = processed_input["attention_mask"][..., :max_position_embeddings]
    #     processed_input["input_ids"] = processed_input["input_ids"][..., :max_position_embeddings]

    txt_features = model.get_text_features(
        processed_input["input_ids"].to(device), processed_input["attention_mask"].to(device)
    )
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity between feature vectors
    score = 100 * (img_features * txt_features).sum(axis=-1)
    return score, len(text)

def _get_clip_model_and_processor(
    model_name_or_path: str = "openai/clip-vit-large-patch14",
) -> Tuple[_CLIPModel, _CLIPProcessor]:
    # from transformers import CLIPModel as _CLIPModel
    # from transformers import CLIPProcessor as _CLIPProcessor

    model = _CLIPModel.from_pretrained(model_name_or_path, use_safetensors=False)
    processor = _CLIPProcessor.from_pretrained(model_name_or_path, use_safetensors=False)
    return model, processor

    # raise ModuleNotFoundError(
    #     "`clip_score` metric requires `transformers` package be installed."
    #     " Either install with `pip install transformers>=4.10.0` or `pip install torchmetrics[multimodal]`."
    # )



# CLIP score metric

class CLIPScore(Metric):
    r"""Calculates `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP Score is a reference free metric that can be used to evaluate the correlation between a generated caption for
    an image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual `CLIP`_ embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``images`` (:class:`~torch.Tensor` or list of tensors): tensor with images feed to the feature extractor with. If
        a single tensor it should have shape ``(N, C, H, W)``. If a list of tensors, each tensor should have shape
        ``(C, H, W)``. ``C`` is the number of channels, ``H`` and ``W`` are the height and width of the image.
    - ``text`` (:class:`~str` or :class:`~list` of :class:`~str`): text to compare with the images, one for each image.

    As output of `forward` and `compute` the metric returns the following output

    - ``clip_score`` (:class:`~torch.Tensor`): float scalar tensor with mean CLIP score over samples

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0

    Example:
        >>> import torch
        >>> from torchmetrics.multimodal.clip_score import CLIPScore
        >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        >>> score = metric(torch.randint(255, (3, 224, 224), generator=torch.manual_seed(42)), "a photo of a cat")
        >>> score.detach()
        tensor(24.4255)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound = 100.0

    score: Tensor
    n_samples: Tensor
    feature_network: str = "model"

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-large-patch14",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = _get_clip_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, images: Union[Tensor, List[Tensor]], text: Union[str, List[str]]) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
            text: Either a single caption or a list of captions

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match

        """
        score, n_samples = _clip_score_update(images, text, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))
