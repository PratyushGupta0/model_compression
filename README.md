# model_compression

The knowledge distillation performed here is based on https://arxiv.org/abs/1503.02531. The idea is to take a large model with a huge number of parameters and try to extract the information contained in it into a smaller model. The dataset used here is the MNIST dataset, whcih is a classification dataset of 60,000 images divided into 10 classes. First we use a large model with 6.8 million parameters for which achieves an accurancy of approximately 0.98. We then distill its information down into a model with 130,000 parameters, which achieves an accurancy of approximately 0.94.

The file "teacher_maker.py" creates and evaluates the original large model. The file "trainer.py" trains the smaller model. The file "evaluator.py" evaluates the smaller model after training.
