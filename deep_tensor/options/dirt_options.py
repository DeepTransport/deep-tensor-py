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
        
        self._verify_method(method)
        
        self.method = method.lower()
        self.max_layers = max_layers
        self.num_samples = num_samples
        self.num_debugs = num_debugs 
        self.defensive = defensive 

        return
    
    def _verify_method(self, method: str) -> None:
        
        if method.lower() in METHODS:
            return
        
        msg = (f"`{method}` is an invalid method. Valid "
               + "methods are: " + ", ".join(METHODS) + ".")
        raise Exception(msg)