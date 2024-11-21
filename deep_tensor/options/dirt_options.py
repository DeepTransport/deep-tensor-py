_DEFAULT_METHOD = "aratio"
_METHODS = ["eratio", "aratio"]
_DEFAULT_MAX_LAYERS = 50
_DEFAULT_NUM_SAMPLES = 1000
_DEFAULT_NUM_DEBUGS = 1000
_DEFAULT_TAU = 1e-8


class DIRTOptions():

    def __init__(
        self, 
        method: str=_DEFAULT_METHOD,
        max_layers: int=_DEFAULT_MAX_LAYERS,
        num_samples: int=_DEFAULT_NUM_SAMPLES,
        num_debugs: int=_DEFAULT_NUM_DEBUGS,
        defensive: float=_DEFAULT_TAU
    ):
        
        self._validate_method(method)
        
        self.method = method.lower()
        self.max_layers = max_layers
        self.num_samples = num_samples
        self.num_debugs = num_debugs 
        self.defensive = defensive 

        return
    
    def _validate_method(self, method: str) -> None:
        
        if method.lower() in _METHODS:
            return
        
        msg = (f"`{method}` is an invalid method. Valid "
               + "methods are: " + ", ".join(_METHODS) + ".")
        raise Exception(msg)