# GAN fingerprint
A deepfake video detection using GAN fingerprint.

We download and use the FaceForensics++ dataset as our dataset. With `preprocess.py`, we extract frames from videos and crop face images from frames using MTCNN. There are six categories: Original, Deepfake, Face2Face, FaceShifter, FaceSwap and NeuralTextures. Presently, each original or manipulated face dataset we use contains 1000 face images.

## Marra's Do GANs leave artificial fingerprints
We try to implement Marra's proposed fingerprint.

We use 900 images of each face set to generate "Marra's GAN fingerprint". We use Guassian filter as denoising filter to get residual. Six categories of "Marra's GAN fingerprint" can be shown as below.

<div align=center>
<img width="120" height="120" src="./README/Figure 2020-12-11 141325 (0).png" >
<img width="120" height="120" src="./README/Figure 2020-12-11 141325 (1).png" >
<img width="120" height="120" src="./README/Figure 2020-12-11 141325 (2).png" >
<img width="120" height="120" src="./README/Figure 2020-12-11 141325 (3).png" >
<img width="120" height="120" src="./README/Figure 2020-12-11 141325 (4).png" >
<img width="120" height="120" src="./README/Figure 2020-12-11 141325 (5).png" >
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

We try to implement Wang's proposed CNN fingerprint classifier.

We first write our own code to reproduce his work. We use his proposed ResNet50 and his trained weights. Then, we test his network performance on his proposed test set. Some results can be shown as below.

```
Test on .\dataset\test\biggan.
Real accuracy:99.25%, Fake accuracy:19.65%, Accuracy:59.45%,Average precision:90.39%.

Test on .\dataset\test\gaugan.
Real accuracy:99.44%, Fake accuracy:66.26%, Accuracy:82.85%,Average precision:98.77%.

Test on .\dataset\test\deepfake.
Real accuracy:99.48%, Fake accuracy:1.74%, Accuracy:50.69%,Average precision:60.18%.
```
The result shows as the same as their paper. However, we can see the Fake accuracy is low, especially in Deepfake manipulation.
Then, we found that the classification threshold in his project code is 0.5 in every category. If we change the threshold to 0.000001, the result will become as below.
```
Test on: .\dataset\test\biggan
Real accuracy:84.00%, Fake accuracy:85.00%, Accuracy:84.50%, Average precision:90.39%.

Test on: .\dataset\test\deepfake
Real accuracy:52.75%, Fake accuracy:60.19%, Accuracy:56.47%, Average precision:60.18%.
```
The best classification threshold will change in each CNN manipulation category, which will affect the accuracies. The average precision can be a more stable metrics to evaluate the network performance. However, the average precision in Deepfake detection is low. Even if we change the classification threshold, it doesn't make sence to improve accuracy because the real and fake output entangle with each other.

<div align=center><img src="./README/Figure 2020-12-18 161525.png" ></div>

Then, we evaluate his network on our FaceForensics++ dataset. Results can show as below.
```
Test on: ../Dataset/Deepfake
Real accuracy:26.40%, Fake accuracy:78.60%, Accuracy:52.50%, Average precision:52.54%.

Test on: ../Dataset/Face2Face
Real accuracy:26.40%, Fake accuracy:80.30%, Accuracy:53.35%, Average precision:51.62%.

Test on: ../Dataset/FaceShifter
Real accuracy:26.40%, Fake accuracy:91.00%, Accuracy:58.70%, Average precision:63.45%.

Test on: ../Dataset/FaceSwap
Real accuracy:26.40%, Fake accuracy:79.30%, Accuracy:52.85%, Average precision:51.12%.

Test on: ../Dataset/NeuralTextures
Real accuracy:26.40%, Fake accuracy:77.60%, Accuracy:52.00%, Average precision:52.82%.
```
We can see it doesn't work well on deepfake manipulation detection.

Then, We tried to re-train the whole network or just re-train the last linear layer of the network with our FaceForensics++ dataset. We used lots of time to fine-tune their network, but the result is not so good, even worse.

## Yu's Attributing Fake Images to GANs - Learning and Analyzing GAN Fingerprints