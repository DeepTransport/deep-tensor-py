from .bridge import Bridge 
from .single_layer import SingleLayer, SavedSingleLayer
from .tempering import Tempering, SavedTempering


BRIDGE2NAME = {
    SingleLayer: "SingleLayer",
    Tempering: "Tempering"
}

NAME2SAVEBRIDGE = {
    "Tempering": SavedTempering,
    "SingleLayer": SavedSingleLayer
}