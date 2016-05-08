#!/usr/bin/python

import sys

highestIndSale = 0
oldKey = None

# Loop around the data
# It will be in the format key\tval
# Where key is the store name, val is the sale amount
#
# All the sales for a particular store will be presented,
# then the key will change and we'll be dealing with the next store

for line in sys.stdin:
    data_mapped = line.strip().split('\t')
    
    if len(data_mapped) != 2:
       continue # something has gone wrong

    thisKey, thisSale = data_mapped

    if oldKey and (oldKey != thisKey):
       print '{0}\t{1}'.format(oldKey, highestIndSale)
       highestIndSale = 0

    oldKey = thisKey
    if highestIndSale < float(thisSale):
       highestIndSale = float(thisSale)

if oldKey != None:
   print '{0}\t{1}'.format(oldKey, highestIndSale)
