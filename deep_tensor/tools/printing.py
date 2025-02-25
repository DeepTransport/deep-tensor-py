GREEN  = "\033[32m"
BLUE = "\033[94m"
END_COLOUR = "\033[0m"


def cross_info(msg):
    print(f"{BLUE}[TT-X]{END_COLOUR} {msg}")

def dirt_info(msg):
    print(f"{GREEN}[DIRT]{END_COLOUR} {msg}")