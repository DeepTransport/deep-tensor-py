from typing import List


def verify_method(method: str, accepted_methods: List[str]) -> None:
        
    if method in accepted_methods:
        return 
    
    msg = (f"Method '{method}' not recognised. Expected one of: " 
            + ", ".join(accepted_methods) + ".")
    raise Exception(msg)