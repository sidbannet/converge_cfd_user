"""
Tool kit for plotting.

@author: siddhartha.banerjee
"""


def get_port_grid_lines(
    ax=None,
    rslt=None,
    linewidth: int = 3,
) -> None:
    """Grid lines for port events in time series plots."""
    xgl = ax.xaxis.get_gridlines()
    xgl[rslt.itick_ipo].set_color('b')
    xgl[rslt.itick_ipo].set_linestyle('--')
    xgl[rslt.itick_ipo].set_linewidth(linewidth)
    xgl[rslt.itick_epo].set_color('r')
    xgl[rslt.itick_epo].set_linestyle('--')
    xgl[rslt.itick_epo].set_linewidth(linewidth)
    xgl[rslt.itick_ipc].set_color('b')
    xgl[rslt.itick_ipc].set_linestyle('-')
    xgl[rslt.itick_ipc].set_linewidth(linewidth)
    xgl[rslt.itick_epc].set_color('r')
    xgl[rslt.itick_epc].set_linestyle('-')
    xgl[rslt.itick_epc].set_linewidth(linewidth)
