'''             CREATE LANGUAGE             '''
# This file creates an entire language with all stages given an input direcectory

import sylconstr
import wordconstr
import soundchange
import sys
import os

# Special error message for nonexistent directories
class DirNotFound(Exception):
    pass

# Goes through each creation step inputted
def createLanguage(swc,direct,syls,wordcount):
    if 's' in swc:
        sylconstr.constructSyls(direct)
    if 'w' in swc:
        wordconstr.makeWords(direct,syls,wordcount)
    if 'c' in swc:
        soundchange.allChanges(direct)

# This program is meant to be run as a main program, so it has a __main__ function.
# Explanation and instructions in the README
if __name__ == "__main__":
    arguments = sys.argv[1:] # gets arguments as a list
    abort = False
    if len(arguments) > 1:
        swc = arguments[0]
        direct = arguments[1]
    else:
        print("Error: Not enough arguments. Please enter at least syllable/word/change (swc) mode and language directory.")
        sys.exit()
    if direct in os.listdir("."):
        if 'inputs' not in os.listdir(direct):
            print("Error:",direct,"does not have an inputs directory.")
            abort = True
        if 'outputs' not in os.listdir(direct):
            print("Error:",direct,"does not have an outputs directory.")
            abort = True
    else:
        print("Error:",direct,"does not exist.")
        sys.exit()
    if not abort:
        syls = 3
        if len(arguments) > 2:
            syls = int(sys.argv[3])
        wordcount = 10000
        if len(arguments) > 3:
            wordcount = int(sys.argv[4])
        createLanguage(swc,direct,syls,wordcount)