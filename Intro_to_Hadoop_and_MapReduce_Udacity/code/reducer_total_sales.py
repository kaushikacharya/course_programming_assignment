#!/usr/bin/python

import sys

countTotalSales = 0
valTotalSales = 0

# Loop around the data
# It will be in the format key\tval
# Where key is the store name, val is the sale amount

for line in sys.stdin:
    data = line.strip().split('\t')
    
    if len(data) != 2:
        continue # some issue with data
        
    thisKey, thisSale = data
    
    countTotalSales += 1
    valTotalSales += float(thisSale)
    
print 'Total number of sales: {0}'.format(countTotalSales)
print 'Total value of sales: {0}'.format(valTotalSales)
