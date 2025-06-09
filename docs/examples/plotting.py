def add_arrows(ax) -> None:
    """Remove the top spine and right spine from a plot, and adds 
    arrows to the ends of the bottom spine and left spine.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(1, 0, ">k", markersize=6, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", markersize=6, transform=ax.transAxes, clip_on=False)
    return