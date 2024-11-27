import abc


class ApproxOptions(abc.ABC):
    
    def _verify_method(
        self,
        method: str,
        accepted_methods: str
    ) -> None:
        
        if method in accepted_methods:
            return 
        
        msg = (f"Method '{method}' not recognised. Expected one of " 
                + ", ".join(accepted_methods) + ".")
        raise Exception(msg)