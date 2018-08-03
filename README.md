# pix2pix

A Tensorflow implementation of the paper 'Image-to-Image Translation with Conditional Adversarial Nets' by Isola, et al.

## Generator Architecture

U-net, and encoder-decoder with skip connections.

## Loss Functions

Loss_g = L1_loss (high level features) + GAN_loss (style/texture)

### L1 Loss

* We incorporate an L1 component in the Generator's loss function
    * (expand) Why is GAN + L1 better than GAN?

* L1 or L2 losses are well known to produce blurry images, because they "average" the plausible outputs
    * L1 and L2 losses are direct measures of the pixel-by-pixel difference between the training output y and the generated image. 
    * Of the two, L1 is slightly less blurry (why?)
    * For this reason, we choose to incorporate L1 (over L2).

* Fails to capture low-level features, like texture and style, due to this blurryness
* But, it can capture high-level features (e.g., there should be a cat here!) (is this true?)

### GAN Loss

Restricted to only capture low-level features (like texture and style).

* PatchGAN
    * Tries to classify if each NxN portion of an image is real or fake
    * Averages the output probabilities for all patches to produce one number
    * We use N = 70
    * Assumes non-overlapping images are independent of each other

### Expected Behavior

* By default, we heavily weight toward L1_loss, by a factor of 100. 
    * So, we would expect L1 to dominate training until it is nearly optimized.
    * Then, the GAN factor would take over.
    * The Discriminator quickly trains, is this going to be a problem?
        * It has been an issue for other types of GANs, because the gradients become too random to be of much use.
        * However, since this discriminator is only learning texture/style, maybe its simple enough that its ok.


## Acknowledgements

* [Image-to-Image Translation with Conditional Adversarial Nets](https://arxiv.org/pdf/1611.07004.pdf)

* [Pytorch implementation by Jun-Yan Zhu](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

* [Affinelayer's Tensorflow implementation](https://github.com/affinelayer/pix2pix-tensorflow)
