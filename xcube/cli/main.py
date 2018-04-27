import argparse
import sys
from typing import List, Optional

from xcube.version import __version__ as version


def main(args: Optional[List[str]] = None):
    args = args or sys.argv[1:]
    parser = argparse.ArgumentParser(description='xcube command-line interface')
    parser.add_argument('--version', '-V', action='version', version=version)
    parser.parse_args(args)


if __name__ == '__main__':
    main()
