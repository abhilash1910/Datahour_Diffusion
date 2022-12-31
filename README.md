# Diffusion Principles

Session for [Analytics Vidhya Datahour](https://datahack.analyticsvidhya.com/contest/datahour-diffusion-models-for-generative-arts/?utm_source=social&utm_medium=twitter&utm_campaign=datahour) on Diffusion principles. 

Youtube recording of the session- https://www.youtube.com/watch?v=DIdbsfzK7IU 

<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png">

Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data). The concept of diffusion was first used by [Sohl etal](https://arxiv.org/abs/1503.03585)

In a bit more detail for images, the set-up consists of 2 processes:

1. A fixed (or predefined) forward diffusion process q of our choosing, that gradually adds Gaussian noise to an image, until you end up with pure noise

2. A learned reverse denoising diffusion process pθ , where a neural network is trained to gradually denoise an image starting from pure noise, until you end up with an actual image.

<img src="https://huggingface.co/blog/assets/78_annotated-diffusion/diffusion_figure.png">

[Diffusion blog using PT from HF](https://huggingface.co/blog/annotated-diffusion)

<img src="https://jalammar.github.io/images/stable-diffusion/article-Figure3-1-1536x762.png">




## Stable Diffusion in TensorFlow / Keras

A Keras / Tensorflow implementation of Stable Diffusion.

This is a fork of [stable-diffusion-tensorflow](https://github.com/fchollet/stable-diffusion-tensorflow)
created by @Fchollet. The weights were ported from the original implementation.


## Usage

1) Try it out with [this GPU Colab](https://colab.research.google.com/drive/1Nl4Z1WFKnw0FvbS0jFGYNsD4phFZNCZ_?usp=sharing).

2) Using the command line :

```
python text2image.py --prompt="An astronaut riding a horse"
```

3) Using the python interface:

```
pip install git+https://github.com/abhilash1910/Datahour_Diffusion
```

```python
from diffusion_tf.stable_diffusion import Text2Image
from PIL import Image

generator = Text2Image(
    img_height=512,
    img_width=512,
    jit_compile=False,
)
img = generator.generate(
    "An astronaut riding a horse",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
	batch_size=1,
)
Image.fromarray(img[0]).save("output.png")
```

## Example outputs

The following outputs have been generated using the this implementation:

1) *A epic and beautiful rococo werewolf drinking coffee, in a burning coffee shop. ultra-detailed. anime, pixiv, uhd 8k cryengine, octane render*

![a](https://user-images.githubusercontent.com/1890549/190841598-3d0b9bd1-d679-4c8d-bd5e-b1e24397b5c8.png)


2) *Spider-Gwen Gwen-Stacy Skyscraper Pink White Pink-White Spiderman Photo-realistic 4K*

![a](https://user-images.githubusercontent.com/1890549/190841999-689c9c38-ece4-46a0-ad85-f459ec64c5b8.png)


3) *A vision of paradise, Unreal Engine*

![a](https://user-images.githubusercontent.com/1890549/190841886-239406ea-72cb-4570-8f4c-fcd074a7ad7f.png)


## Addition of Linear Attention in Diffusion Transformer

<img src="https://github.com/lucidrains/linear-attention-transformer/blob/master/linear-attention.png?raw=true">

 Attention mechanism that mixes (QKᵀ)V local attention with Q(KᵀV) global attention (scales linearly with respect to sequence length) for efficient long-range language modeling. Linear attention scales linearly with time and hence avoids the quadratic runtime complexity of the local cross attention mechanism. The code for the LinearAttention class (keras.layer) can be found inside [diffusion_model.py](https://github.com/abhilash1910/Datahour_Diffusion/blob/9ed0be57a86b70dad346d3af02a710569e6d823d/diffusion_tf/diffusion_model.py#L37)
 Linear attention code adapted from  [Phil Wang's  torch implementation](https://github.com/lucidrains/linear-attention-transformer) . Can be abstracted as a separate tf.keras.layers.Layer (keras layer) as well.

## References

1) https://github.com/CompVis/stable-diffusion
2) https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
