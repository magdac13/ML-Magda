# Tensor is a generalization of vectors and matrices to potentially higher dimensions.
# Internally, TensorFlow represents tensors as n-dimensional arrays of base data types.
import tensorflow as tf
import numpy as np

# creating tensors:

string = tf.Variable("this is a string", tf.string)
number = tf.Variable(12345, tf.int16)
floating = tf.Variable(12345.6789, tf.float64)


# Rank/Degree of a tensor - these terms mean the number of dimensions in the tensor.

rank1_tensor = tf.Variable(["Test"], tf.string)
rank2_tensor = tf.Variable([["test", "yes"], ["test", "ok"]], tf.string)

# To determine the rank of a tensor, we can use tf.rank.

tf.rank(rank1_tensor)

# Shape of a tensor - these terms mean the shape of the tensor.

rank2_tensor.shape()

# The number of elements of a tensor is the product of the sizes of all its shapes.
# There are often many shapes that have the same number of elements, making it convient to be able to
# change the shape of a tensor.
# The example below shows how to change the shape of a tensor.

tensor1 = tf.ones([1, 2, 3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2, 3, 1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
# this will reshape the tensor to [3,3]

# The numer of elements in the reshaped tensor MUST match the number in the original

# Evaluating a tensor (getting the value of a tensor)

# with tf.Session() as sess:  creates a session using the default graph
#     tensor.eval()    evaluates the tensor

Współczynnik R2, również nazywany współczynnikiem determinacji, określa, jak dobrze model dopasowuje się do danych.

Współczynnik R2 przyjmuje wartości od 0 do 1, gdzie 0 oznacza, że model nie tłumaczy zmienności celu, a 1 oznacza, że model doskonale tłumaczy zmienność celu.

Formalnie, współczynnik R2 obliczany jest jako stosunek wyjaśnionej zmienności (sumy kwadratów różnic między predykcjami a średnią wartością celu) do całkowitej zmienności (sumy kwadratów różnic między wartościami rzeczywistymi a średnią wartością celu). Wartości R2 w zakresie od 0 do 1 oznaczają, jak wiele zmienności celu jest wyjaśnione przez model.

W praktyce, wysoka wartość współczynnika R2 wskazuje, że model dobrze dopasowuje się do danych i przewiduje wartości celu z dużą dokładnością. Jednakże, warto zauważyć, że R2 może być mylący w przypadku, gdy modele różnią się pod względem liczby zmiennych lub zakresów wartości cech. W takich przypadkach, warto skorzystać z innych metryk, takich jak błąd średniokwadratowy (MSE) czy średni błąd absolutny (MAE), aby dokładniej ocenić jakość modelu.


