
"""DQN Class
DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""

# https://github.com/hunkim/ReinforcementZeroToAll
# dqn.py
# to tensorflow 2.1?

import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, input_size: int, output_size: int, input_shape, name: str = 'main') -> None:
        # The meaning of '->' is given in
        # https://stackoverflow.com/questions/14379753/what-does-mean-in-python-function-definitions
        """DQN Agent can

        1) Build network
        2) Predict Q_value given state
        3) Train parameters

        Args:
            input_size (int) : Input dimension
            output_size (int) : Number of discrete actions
            name (str, optional) : TF Graph will be built under this name scope
        """

        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.input_shape = input_shape
        self._build_network()

    def _build_network(self, h_size=16) -> None:
        """DQN Network architecture (simple MLP)

        Args:
            h_size (int, optional) : Hidden layer dimension
            l_rate (float, optional) : Learning rate
        """

        self.model = tf.keras.Sequential([

            tf.keras.layers.Dense(h_size, input_shape=self.input_shape, activation='relu'),
            tf.keras.layers.Dense(self.output_size)
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s,a)

        Args:
            state (np.ndarray) : State array, shape(n, input_dim)

        Returns:
            np.ndarray : Q value array, shape (n, output_dim)
        """

        x = np.reshape(state, [-1, self.input_size])

        return self.model.predict(x)

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and return a result

        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack(np.ndarray): Target Q array, shape (n, output_dim)

        Returns:
            list: First element is loss, second element is a result from train step
        """

        return self.model.fit(x_stack, y_stack)

    def copy_network(self, src):
        self.model.set_weights(src.model.get_weights())



