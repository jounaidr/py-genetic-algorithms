from common import *
import numpy as np
import difflib as diff
import concurrent.futures
from threading import Lock
import time
import string


yomasd = "asdsad"
the_thing = list(string.printable)
print(the_thing)

a = np.array(['a', 'b', 'c'])
dom = np.array(['b', 'a', 'a'])
print(a)
b = "".join(a)
print(b)

yop = np.sum(a == dom)

print(yop)