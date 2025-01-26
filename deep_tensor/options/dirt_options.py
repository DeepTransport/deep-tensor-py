from .verification import verify_method


METHODS = ["eratio", "aratio"]


class DIRTOptions():

    def __init__(
        self, 
        method: str="aratio",
        max_layers: int=50,
        num_samples: int=1000,
        num_debugs: int=1000,
        defensive: float=1e-8
    ):
        
        method = method.lower()
        verify_method(method, METHODS)

        self.method = method
        self.max_layers = max_layers
        self.num_samples = num_samples
        self.num_debugs = num_debugs 
        self.defensive = defensive 
        return