#!/usr/bin/python

import sys

pageHits = 0
oldPage = None

for line in sys.stdin:
    data_mapped = line.strip().split('\t')
    
    if len(data_mapped) != 2:
        continue
    
    thisPage,thisCount = data_mapped
    
    if oldPage and (oldPage != thisPage):
        print '{0}\t{1}'.format(oldPage, pageHits)
        pageHits = 0 # re-initializing
        
    pageHits += float(thisCount)
    oldPage = thisPage
    
if oldPage != None:
    print '{0}\t{1}'.format(oldPage, pageHits)
