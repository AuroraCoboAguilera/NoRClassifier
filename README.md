# NoRClassifier

This repository includes a notebook with a basic example of how to incorporate a stochastic layer in a classification network applied in FMNIST dataset.
The stochastic layer is defined through a GMVAE model and the overall training has 4 steps:

1. Pre-training the classification network in FMNIST.
2. Traininig a GMVAE with the hidden vectors after the layer 1 from the pre-trained classifier.
3. Integrating the GMVAE as a stochastic layer between layers 1 and 2 form the initial classifier.
4. Fine-tuning layers 2-9 in a new architecture build with the stochastic layer.
