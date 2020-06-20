# -*- coding: utf-8 -*-
"""

Created on Thursday June 18 17:38:40 2020
@author: Krishna

This script will combine all the */txt files into a
single file for data analysis

"""

import glob

read_files = glob.glob("Health-Tweets/*.txt")

with open("result.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
outfile.close()