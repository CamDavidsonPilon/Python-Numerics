

#sample a file by taking every nth item
# usage:
# $ python sample.py n myfile.txt mysampledfile.txt
#
#

import sys


def sample(n, infile, outfile):
    
    n = int(n)
    try:
        ifile = open( infile, 'r')
    except e:
        print "Could not open file %s"%infile
        raise e
    
    ofile = open ( outfile, 'w')
    
    i = 0
    for line in ifile.readlines():
        if i%n==0:
            ofile.write(line)
        i+=1



if __name__ == "__main__":
    sample( *sys.argv[1:] )
    print "Completed"