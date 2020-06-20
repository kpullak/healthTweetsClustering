# -*- coding: utf-8 -*-

import glob

read_files = glob.glob("Health-Tweets/*.txt")

with open("result.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
outfile.close()