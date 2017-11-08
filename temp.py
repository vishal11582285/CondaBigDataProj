
from time import sleep
import sys

for i in range(51):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-50s] %d%%" % ('='*i, 2*i))
    sys.stdout.flush()
    sleep(0.25)