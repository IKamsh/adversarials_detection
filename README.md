# adversarials_detection (in development)

repository contains detectors of adversarial examples

to generate attack (FGSM) was used foolbox 2.4.0 <code>attack.py</code> \
<code>adversarials_detection.ipynb</code> demonstrates how to use adversarial detector \
all experiments were performed using VGG architecture <code>vgg.py</code> \
(to train model one could use <code>cifar10training.py</code>) \
Detector algorithms are in <code>detectors.py</code>

### Adversarial examples

Original image: \
![alt text](https://github.com/IKamsh/adversarials_detection/blob/main/img/dog.PNG) \
The same image with small perturbations: \
![alt text](https://github.com/IKamsh/adversarials_detection/blob/main/img/adv_dog.PNG)

### Algorithms

Softmax output of NN for the pictures above:
![alt text](https://github.com/IKamsh/adversarials_detection/blob/main/img/softmax_distribution.PNG)
As we see, "probability" of real class in perturbed image is still significant. It could be \
used to teach binary classificator (0 - real, 1 - adversarial) with softmax as features.
