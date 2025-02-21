from dataclasses import dataclass

from .verification import verify_method


METHODS = ["eratio", "aratio"]


@dataclass
class DIRTOptions():

    method: str = "aratio"
    max_layers: int = 50
    num_samples: int = 1000
    num_debugs: int = 1000
    defensive: float = 1e-8
    verbose: bool = True
        
    def __post_init__(self):
        self.method = self.method.lower()
        verify_method(self.method, METHODS)
        return