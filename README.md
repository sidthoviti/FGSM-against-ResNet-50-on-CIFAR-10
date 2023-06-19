# FGSM-against-ResNet-50-on-CIFAR-10
Fast Gradient Sign Method against ResNet-50 on CIFAR-10 - Adversarial Machine Learning

Read about this project on my blog: [Part 2: Fast Gradient sign Method (FGSM)](https://sidthoviti.com/part-2-fast-gradient-sign-method-fgsm/)

This project tests the robustness of a [fine-tuned ResNet50](https://sidthoviti.com/fine-tuning-resnet50-pretrained-on-imagenet-for-cifar-10/) for CIFAR-10 against FGSM.

The attack success rate and robustness of the model were tested against various perturbation sizes.

![Comparison of Adversarial Accuracies and Attack Success Rates](https://github.com/sidthoviti/FGSM-against-ResNet-50-on-CIFAR-10/assets/96778922/2e9a6e2e-a3f8-43eb-8a95-02f3738b149a)

* After training the model for 60 epochs, it achieved a clean test accuracy of 92.63%.
* The model is slightly overfitting.
* The model was then tested with adversarial examples generated using different epsilon values. With an epsilon of 0.01, the adversarial accuracy dropped to 47.02%, indicating successful attacks on the model.
* As the epsilon value increased, the adversarial accuracy decreased further, reaching 24.62% at an epsilon of 0.5.
* The attack success rate increased with higher epsilon values, ranging from 52.98% at epsilon 0.01 to 75.38% at epsilon 0.5.

![Generated Adversarial Examples](https://github.com/sidthoviti/FGSM-against-ResNet-50-on-CIFAR-10/assets/96778922/49798c05-9ea5-4915-9f58-1b2a0b665820)

* The adversarial images look pixelated at just 0.3 perturbation size as CIFAR-10's input size is 32x32. As observed from the above results, all the randomly selected images are misclassified.
