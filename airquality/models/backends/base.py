from abc import abstractmethod

class ModelWrapper:
    """ Interface for models to interact with optimizers
    """
    @abstractmethod
    def set_coef(self, coefs, indices=None):
        """ Assign coefficients to a fitted model (for evaluating ensembles, etc.)
        """
        raise NotImplementedError
    
    @abstractmethod
    def _regress(self, x, y, **fit_kwargs):
        """ Perform the base-level fitting using the underlying regression algorithm
        """
        raise NotImplementedError
    
    @abstractmethod
    def _make_initial_guess(self, x, y, **fit_kwargs):
        """ Initialize the model for fitting
            Create self.model_, self.coef_, self.indices_
        """
        raise NotImplementedError

    @abstractmethod
    def summary_dataframe(self, feature_names):
        """ Create a summary dataframe for the model
        """
        raise NotImplementedError