class EarlyStopping:
    """
    Implements early stopping to terminate training when the loss
    does not improve after a specified number of epochs.
    """

    def __init__(self, patience=5, min_delta=0.0):
        """
        Initializes EarlyStopping object.

        Args:
        :param patience: Number of epochs to wait after the last improvement.
        :param min_delta: Minimum change in the monitored loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, loss):
        """
        Call method to update the stopping condition based on the
        current loss.

        Args:
        :param loss: The current loss.
        :return: None
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self):
        """
        Resets the internal state for reuse in case training should still
        continue for some special reasons.

        :return: None
        """
        self.counter = 0
        self.early_stop = False