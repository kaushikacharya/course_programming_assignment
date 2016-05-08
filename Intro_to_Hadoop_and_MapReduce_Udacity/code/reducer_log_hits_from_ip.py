#!/usr/bin/python

import sys

IPhits = 0
oldIP = None

for line in sys.stdin:
    data_mapped = line.strip().split('\t')
    
    if len(data_mapped) != 2:
        continue
        
    thisIP,countHit = data_mapped
    
    if oldIP and (oldIP != thisIP):
        print '{0}\t{1}'.format(oldIP, IPhits)
        IPhits = 0 # re-initialize for thisIP
        
    IPhits += float(countHit)
    oldIP = thisIP
    
if oldIP != None:
    print '{0}\t{1}'.format(oldIP, IPhits)
