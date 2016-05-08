#!/usr/bin/python

import sys

for line in sys.stdin:
    data_mapped = line.strip().split()
    
    if len(data_mapped) != 10:
        continue
        
    thisIP = data_mapped[0]
    
    print '{0}\t{1}'.format(thisIP,1)
