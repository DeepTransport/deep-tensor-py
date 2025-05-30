GREEN  = "\033[32m"
BLUE = "\033[94m"
END_COLOUR = "\033[0m"


def als_info(msg, **kwargs):
    print(f"{BLUE}[ALS]{END_COLOUR}  {msg}", **kwargs)

def dirt_info(msg, **kwargs):
    print(f"{GREEN}[DIRT]{END_COLOUR} {msg}", **kwargs)