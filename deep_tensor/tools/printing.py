GREEN  = "\033[32m"
BLUE = "\033[94m"
END_COLOUR = "\033[0m"


def als_info(msg: str, **kwargs):
    print(f"{BLUE}[ALS]{END_COLOUR}  {msg}", **kwargs)


def dirt_info(msg: str, **kwargs):
    print(f"{GREEN}[DIRT]{END_COLOUR} {msg}", **kwargs)


def format_time(t: float) -> str:
    if t < 60: 
        t_formatted = f"{t:.2f} secs"
    elif t < 3600:
        t_formatted = f"{t/60.0:2.2f} mins"
    else:
        t_formatted = f"{t/3600.0:2.2f} hours"
    return t_formatted