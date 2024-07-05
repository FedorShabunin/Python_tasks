#!/usr/bin/env python3
# coding: utf-8

from plotter import plot
from data_runner import run, default_nvalues


def main(argv: list):
    
    density = int(argv[2]) if argv[2] is not None else 10
    nvalues = default_nvalues
    #nvalues = (10,)
    run(nvalues = nvalues, density = density)
    plot(nvalues = nvalues)

if __name__ == '__main__':
    import sys
    main(sys.argv)
