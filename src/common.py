import numpy

""" LOADS TICTAC FINAL AND SINGLE DATA
    INPUTS:
        1 = SINGLE DATA
        0 = FINAL DATA
    RETURNS:
        X: Training Feature Vector
        Y: Labels
"""
def load_tictac_final_single(file):
    if(file):
        A = numpy.loadtxt("../datasets-part1/tictac_single.txt")
    else:
        A = numpy.loadtxt("../datasets-part1/tictac_final.txt")

    X = A[:,:9]
    Y = A[:,9:]
    return X,Y

""" LOADS TICTAC MULTI DATA
    INPUTS:
        NA
    RETURNS:
        X: Training Feature Vector
        Y: Labels
"""
def load_tictac_multi():
    A = numpy.loadtxt("../datasets-part1/tictac_multi.txt")
    X = A[:,:9]
    Y = A[:,9:]
    return X,Y

""" WRITES INFO TO FILE
    INPUTS:
        name: name of file
        lines: info to write
    RETURNS:
        NA
"""
def file_write(name, lines):
    with open(name,'w') as f:
        f.writelines(lines)