# Transformation Complexity

PyTorch implementation of the paper "Towards Theoretical Analysis of Transformation Complexity of ReLU DNNs"

## Requirements

- Python 3


## Usage

**Diagnose the transformation complexity of a trained DNN**

*TODO*

**Correlation between the complexity and the disentanglement**

*TODO*

**Penalize the transformation complexity**

You can run the following code to penalize the transformation complexity. Please specify the target gating layers and their corresponding energy functions. You can also tune the coefficient $\lambda$ and other hyper-parameters. (Note that in the code, the value of $\lambda$ is set as its log value.)

~~~shell
python3 main_penalize_transformation.py --dataset=cifar10 --arch=resmlp10 \
        --penalize-layers=layers.5.act,layers.6.act,layers.7.act,layers.8.act \
        --energy-functions=E_3072d,E_3072d,E_3072d,E_3072d \
        --n-channels=3072,3072,3072,3072 --loss_lambda=-3.0
~~~

To evaluate the transformation complexity of the above trained DNN, you can add the `--evaluate` flag after your command.

