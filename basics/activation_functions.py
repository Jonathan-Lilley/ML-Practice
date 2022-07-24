import numpy as np


class Activation:

    def __init__(self, activ_name):
        self.activation_name = activ_name
        activations = {
            "sigmoid": self._sigmoid(),
            "softmax": self._softmax(),
        }
        self.activation, self.derivative = \
            activations.get(activ_name, activations["sigmoid"])

    def _sigmoid(self):
        def activation(x):
            return 1 / (1 + np.power(np.e, -x))

        def derivative(x):
            return activation(x) * (1 - activation(x))
        return activation, derivative

    def _tanh(self):
        def activation():
            pass

        def derivative():
            pass
        return activation, derivative

    def _relu(self):
        def activation():
            pass

        def derivative():
            pass
        return activation, derivative

    def _softmax(self):
        def activation(x):
            shiftx = x - np.max(x)
            exps = np.exp(shiftx)
            return exps / np.sum(exps)

        def derivative(x):
            """ Derivative of softmax

            The math:
            dSi/daj = Sj(1{i == j} - Si)
            This can be simplified by taking the dot product of S and S.T,
            creating a square matrix, then summing the row values (Si) and
            adding the activation (Sj) to each row sum.

                    { S1-S1*S1  0-S1*S2  0-S1*S3

                      0-S2*S1  S2-S2*S2  0-S2*S3

                      0-S3*S1  0-S3*S2  S3-S3*S3 }

                    { S1-ΣS1*Sj, S2-ΣS2*Sj, S3-ΣS3*Sj }

            """
            activated = activation(x)
            S = np.sum(np.dot(activated.reshape(-1, 1),
                              activated.reshape(1, -1)), axis=1)
            out = activated - S
            return out
        return activation, derivative
