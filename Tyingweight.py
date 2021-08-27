from torch import nn


# https://zhuanlan.zhihu.com/p/358599463
# introduce nn.functional.pad

def tie_weights(output_embeddings, input_embeddings):
    """Tie or clone module weights depending of whether we are using TorchScript or not"""
    output_embeddings.weight = input_embeddings.weight

    if getattr(output_embeddings, "bias", None) is not None:
        output_embeddings.bias.data = nn.functional.pad(
            output_embeddings.bias.data,
            (
                0,
                output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )
    if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
        output_embeddings.out_features = input_embeddings.num_embeddings


"""
Often, we want to share parameters across multiple layers. Let us see how to do this elegantly. 
In the following we allocate a dense layer and then use its parameters specifically to set those of another layer.

# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                    nn.ReLU(), nn.Linear(8, 1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])

outputs:
tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])

This example shows that the parameters of the second and third layer are tied. They are not just equal, 
they are represented by the same exact tensor. Thus, if we change one of the parameters, the other one changes, too. 
You might wonder, when parameters are tied what happens to the gradients? Since the model parameters contain gradients, 
the gradients of the second hidden layer and the third hidden layer are added together during backpropagation.
"""
