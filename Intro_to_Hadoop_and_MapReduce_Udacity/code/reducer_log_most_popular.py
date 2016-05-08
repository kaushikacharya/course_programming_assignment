#!/usr/bin/python

import sys

oldPage = None
oldHits = 0
mostPopularPage = None
mostPopularHits = 0

for line in sys.stdin:
    data_mapped = line.strip().split('\t')
    
    if len(data_mapped) != 2:
        continue
        
    thisPage, thisCnt = data_mapped
    
    if oldPage and (oldPage != thisPage):
        if mostPopularHits < oldHits:
            mostPopularHits = oldHits
            mostPopularPage = oldPage
        
        oldHits = 0
        
    oldHits += float(thisCnt)
    oldPage = thisPage
    
if oldPage != None:
    if mostPopularHits < oldHits:
        mostPopularHits = oldHits
        mostPopularPage = oldPage
        
    print '{0}\t{1}'.format(mostPopularPage, mostPopularHits)
    
        
    
