import sys

from rich import print as rprint
from rich.text import Text

from llmagent.utils.configuration import settings
from llmagent.utils.constants import Colors


def print_long_text(
    color: str, style: str, preamble: str, text: str, chars: int = None
):
    if chars is not None:
        text = " ".join(text.split())
        text = text[:chars] + "..." + text[-chars:] if len(text) > 2 * chars else text
    styled_text = Text(text, style=style)
    rprint(f"[{color}]{preamble} {styled_text}")


def show_if_debug(
    text: str,
    preamble: str,
    chars: int = None,
    color: str = "red",
    style: str = "italic",
):
    if settings.debug:
        print_long_text(color, style, preamble, text, chars)


class PrintColored:
    """Context to temporarily print in a desired color"""

    def __init__(self, color: str):
        self.color = color

    def __enter__(self):
        sys.stdout.write(self.color)
        sys.stdout.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(Colors().RESET)


# code to show a spinner while waiting for a long-running process

# from tqdm import trange
#
# def black_box():
# # ...code for your black box function here...
#
# # Use trange to display a spinner
# for i in trange(1, desc="Processing", unit="call", ncols=75, ascii=True, leave=False):
#     result = black_box()
#
# # Update progress bar to completion
# print("Done!")
