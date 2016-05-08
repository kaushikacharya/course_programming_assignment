#!/usr/bin/python

import sys

for line in sys.stdin:
    data_mapped = line.strip().split()
    
    if len(data_mapped) != 10:
        continue # log sentence isn't for page hit
        
    thisIP = data_mapped[0]
    thisPage = data_mapped[6]
    
    print '{0}\t{1}'.format(thisPage, 1)
