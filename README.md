# GAN fingerprint
A deepfake video detection using GAN fingerprint.

We download and use the FaceForensics++ dataset as our dataset. With `preprocess.py`, we extract frames from videos and crop face images from frames using MTCNN. There are six categories: Original, Deepfake, Face2Face, FaceShifter, FaceSwap and NeuralTextures. Presently, each original or manipulated face dataset we use contains 1000 face images.

## Marra's Do GANs leave artificial fingerprints
We try to implement Marra's proposed fingerprint.

We use 900 images of each face set to generate "Marra's GAN fingerprint". We use Guassian filter as denoising filter to get residual. Six categories of "Marra's GAN fingerprint" can be shown as below.

<div align=center>
<img width="100" height="100" src="./README/Figure 2020-12-11 141325 (0).png" >
<img width="100" height="100" src="./README/Figure 2020-12-11 141325 (1).png" >
<img width="100" height="100" src="./README/Figure 2020-12-11 141325 (2).png" >
<img width="100" height="100" src="./README/Figure 2020-12-11 141325 (3).png" >
<img width="100" height="100" src="./README/Figure 2020-12-11 141325 (4).png" >
<img width="100" height="100" src="./README/Figure 2020-12-11 141325 (5).png" >
</div>

Then, we use the rest 100 images of each face set to test whether "Marra's GAN fingerprint" works. Our test method follows his paper. First, calculate the correlation index between the residual of test image and the six specific fingerprints, then choose the highest correlation index one as pridicted manipulated method. The result can be shown as below.

```
The test accuracy in Original_c40_faces is 13.00%
The test accuracy in Deepfake_c40_faces is 40.00%
The test accuracy in Face2Face_c40_faces is 31.00%
The test accuracy in FaceShifter_c40_faces is 27.00%
The test accuracy in FaceSwap_c40_faces is 25.00%
The test accuracy in NeuralTextures_c40_faces is 9.00%
```

It seems that Marra's method is not so good to deal with the deepfake detection because his fingerprint is calculated on whole GAN synthetic image. The deepfake situation may include extra manipulations like compression, blending and some other image processing procedures. Besides, the denoising method we use may differ from his. We will continue to analyse it...

## Wang's CNN-generated images are surprisingly easy to spot... for now

Will be updated soon.