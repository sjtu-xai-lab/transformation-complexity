# Transformation Complexity

PyTorch implementation of the paper ["Towards Theoretical Analysis of Transformation Complexity of ReLU DNNs"](https://arxiv.org/abs/2205.01940)

## Requirements

- Python 3
- PyTorch==1.8.0


## Usage

**Diagnose the transformation complexity of a trained DNN**

You can run the following code to evaluate the transformation complexity of a network. Please manually extract the gating states in the model, and specify the model root as follows.

~~~shell
python3 main_evaluate_complexity.py --dataset=mnist --arch=mlp_mnist \
        --epochs=501 --eval-interval=5 \
        --model-root=./saved-models/dataset=mnist_model=mlp
~~~

**Penalize the transformation complexity**

You can run the following code to penalize the transformation complexity. Please specify the target gating layers and their corresponding energy functions. You can also tune the coefficient $\lambda$ and other hyper-parameters. (Note that in the code, the value of $\lambda$ is set as its log value.)

~~~shell
python3 main_penalize_transformation.py --dataset=cifar10 --arch=resmlp10 \
        --penalize-layers=layers.5.act,layers.6.act,layers.7.act,layers.8.act \
        --energy-functions=E_3072d,E_3072d,E_3072d,E_3072d \
        --n-channels=3072,3072,3072,3072 --loss_lambda=-3.0
~~~

To evaluate the transformation complexity of the above trained DNN, you can add the `--evaluate` flag after your command.

**Demos**

We have also provided some demos to reproduce results in the paper.

1. Evaluation of transformation complexity. See [`notebooks/evaluate_transformation_complexity.ipynb`](notebooks/evaluate_transformation_complexity.ipynb)
2. The utility of the complexity loss. See [`notebooks/penalize_transformation_complexity.ipynb`](notebooks/penalize_transformation_complexity.ipynb)
