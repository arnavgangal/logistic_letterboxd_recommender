import numpy as np

from surprise import AlgoBase
from surprise import PredictionImpossible
import time


class LogFactorizer(AlgoBase):
    def __init__(self, n_factors=10, n_epochs=50, lr_users=0.0001, lr_items=0.01, reg_term=0.3, verbose=False):
        """
        Initialize the Logistic Matrix Factorization model
        :param n_factors: dimension of latent factors
        :param n_epochs: number of training epochs
        :param lr_users: learning rate for users
        :param lr_items: learning rate for items
        :param reg_term: regularization coefficient
        """
        # Call the constructor of the parent class (AlgoBase)
        AlgoBase.__init__(self)

        # Add fields
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_users = lr_users
        self.lr_items = lr_items
        self.reg_term = reg_term
        self.verbose = verbose

    def fit(self, trainset):
        # Call superclass fit method
        AlgoBase.fit(self, trainset)

        # Initialize user and item latent matrices randomly
        self.m = trainset.n_users
        print("The trainset has ", self.m, " users")
        self.n = trainset.n_items
        print("The trainset has ", self.n, " items")
        self.p_matrix = np.random.randn(self.m, self.n_factors)
        self.q_matrix = np.random.randn(self.n, self.n_factors)

        # Set alpha
        self.alpha = self.calc_alpha(trainset)

        # Call the training method
        self.sgd(trainset)

        return self

    def sgd(self, trainset):
        """ Implementation of Alternating Gradient Descent

        :param trainset: TrainSet object containing users and items
        """
        for train_step in range(self.n_epochs):
            t0 = time.time()
            print("Starting epoch ", train_step+1)
            # Fix Q, update P
            for u in trainset.all_users():
                t00 = time.time()
                # Use a set to create the mask
                user_items = set(tup[0] for tup in self.trainset.ur[u])
                r_ui_list = np.array([1 if i in user_items else 0 for i in range(self.n)])

                alpha_term = self.alpha * r_ui_list[:, np.newaxis] * self.q_matrix
                epq = np.exp(self.q_matrix.dot(self.p_matrix[u]))
                epq = epq[:, np.newaxis]
                grads = (- alpha_term + ((self.q_matrix + alpha_term) * epq) / (1 + epq)).sum(axis=0)
                # Add a regularizing term
                grads += 2 * self.reg_term * self.p_matrix[u]
                self.p_matrix[u] -= self.lr_users * grads
                t10 = time.time()
                if (u % 10 == 0) and self.verbose:
                    print(f"User {u} in epoch {train_step+1} finished in {t10 - t00:.2f} seconds")

            # Fix P, update Q
            for i in trainset.all_items():
                t00 = time.time()
                item_users = set(tup[0] for tup in self.trainset.ir[i])
                r_ui_list = np.array([1 if u in item_users else 0 for u in range(self.m)])
                alpha_term = self.alpha * r_ui_list[:, np.newaxis] * self.p_matrix
                epq = np.exp(self.p_matrix.dot(self.q_matrix[i]))
                epq = epq[:, np.newaxis]
                grads = (- alpha_term + ((self.p_matrix + alpha_term) * epq) / (1 + epq)).sum(axis=0)
                grads += 2 * self.reg_term * self.q_matrix[i]
                self.q_matrix[i] -= self.lr_items * grads
                t10 = time.time()
                if (i % 1000 == 0) and self.verbose:
                    print(f"Item {i} in epoch {train_step+1} finished in {t10 - t00:.2f} seconds")

            # Consider adding a diagnostic tool to trace the loss on logits, and check training success

            t1 = time.time()
            print("Epoch %i finished in %f seconds" % (train_step+1, t1-t0))

    def calc_alpha(self, trainset):
        """ Get the value of alpha
        :param trainset: trainset of data
        :return:
        """
        numerator = (trainset.n_users * trainset.n_items) - trainset.n_ratings
        return numerator / trainset.n_ratings

    def estimate(self, user, item):
        """ Returns the predicted value for a user and item pair
        :param user: inner id of user
        :param item: inner id of item
        :return: Predicted score for user-item pair
        """

        known_user = self.trainset.knows_user(user)
        known_item = self.trainset.knows_item(item)

        if not (known_user and known_item):
            raise PredictionImpossible('User or item is unknown.')
        else:
            est = self._sigmoid(np.dot(self.q_matrix[item], self.p_matrix[user]))

        return est

    def _sigmoid(self, x):
        """
        Sigmoid activation (numerically stable)
        """
        return np.exp(-np.logaddexp(0, -x))
