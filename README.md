# Video Colorization using Deep Neural Networks

We train three different models to colorize a grayscale video.

- ColorizationNet -> A pretrained resnet with upsampling as generator, PATCHGan as discrimainator, and trained on BCE, MSE, L1, and Perceptual Loss.
- Colorization with U-Net -> An U-Net encoder-decoder acts as generator, PATCHGan as discrimainator, and model is trained using NOGan methodlogy where generator and discriminator are pre-trained separately before adversarial training is done.
- Colorization with U-Net and Wasserstein Loss -> Addition of Wasserstein Loss on second model.

A output sample is store in output folder with results from different models. Please refer to report file [Video Colorization using Deep Neural Networks.pdf](https://github.com/hanzalah21027/Video-Colorization-using-Deep-Neural-Networks/blob/main/Video%20Colorization%20using%20Deep%20Neural%20Networks.pdf) for detailed explanation on the architecture, training regime, and results of the models created.

The Train file for each model is in their respective folder. Once model are trained, the test code for all of them can be found in test.ipynb.

## Results

### GrayScale Video
https://user-images.githubusercontent.com/88739322/219767886-f86d0fa9-d5f1-4ffb-b69e-f98d6a6e9597.mp4


### Coloured Video
https://user-images.githubusercontent.com/88739322/219769127-3fdb5117-5f7f-4c85-917a-9ba605be8264.mp4

