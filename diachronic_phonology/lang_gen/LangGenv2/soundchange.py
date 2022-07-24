#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''                 SOUND CHANGE                    '''

import IPA, helpers, os, codecs, sys

reload(sys)
sys.setdefaultencoding('utf-8')

# Checks to see if two parts of a string are identical
# Part of findSubstring
# Input i is initial index
def checkIdent(string, substring, i):
    j = 0
    while j < len(substring) and i + j < len(string):
        if string[i + j] == substring[j]:
            j += 1
            if j == len(substring):
                return True
        else:
            return False

# Finds index of a substring within a string
def findSubstring(string, substring):
    i = 0
    while i < len(string):
        if string[i] == substring[0]:
            ident = checkIdent(string, substring, i)
            if ident:
                return i
        i += 1
    return -1

# Replaces all instances of substring with replstr (replacement string) in string
def replaceSubstring(string, substring, replstr):
    i = findSubstring(string, substring)
    while i != -1:
        string = string[:i] + replstr + string[i+len(substring):]
        i = findSubstring(string,substring)
    return string

# Replaces alpha features
def replaceAlpha(rule,IPA_info):
    pmvhbr = IPA_info[4] # places, manners, voicings, heights, backnesses, roundings
    # Sets up arrays and bool max
    present = []
    rules = []
    replacements = []
    indexes = []
    maxindexes = []
    max = False
    # Checks if there is an alpha value in the rule
    if findSubstring(rule, "a") == -1: # If not, add the rule as is
        rules.append(rule)
    # If so, generate all alpha replacement rules
    # replacements is an array of arrays where the subarrays are the place/manner/voicing/etc value arrays
    # present indicates if a particular alpha value is present
    else:
        if findSubstring(rule, "PLACE:a") != -1:
            replacements.append(pmvhbr[0])
            present.append("PLACE:a")
        if findSubstring(rule, "MANNER:a") != -1:
            replacements.append(pmvhbr[1])
            present.append("MANNER:a")
        if findSubstring(rule, "VOICING:a") != -1:
            replacements.append(pmvhbr[2])
            present.append("VOICING:a")
        if findSubstring(rule, "HEIGHT:a") != -1:
            replacements.append(pmvhbr[3])
            present.append("HEIGHT:a")
        if findSubstring(rule, "BACKNESS:a") != -1:
            replacements.append(pmvhbr[4])
            present.append("BACKNESS:a")
        if findSubstring(rule, "ROUNDING:a") != -1:
            replacements.append(pmvhbr[5])
            present.append("ROUNDING:a")
        # Sets up indexes and max index values
        for aspect in replacements:
            indexes.append(0)
            maxindexes.append(len(aspect)-1)
        # Goes through and replaces all alpha instances in all combinations
        while not max:
            # initializes newrule as rule so newrule can have its alpha features replaced
            newrule = rule
            # Replaces all alpha values present with current iteration's values
            for aspect in range(len(replacements)):
                newrule = replaceSubstring(newrule,present[aspect],replacements[aspect][indexes[aspect]])
            rules.append(newrule)
            indexes, max = helpers.increment(indexes,maxindexes) # Next iteration
    return rules

# Expands environment rules to generate all possible environment rules
def generateEnvirons(environment,IPA_info):
    IPA_info = IPA_info[:4]
    eles = environment.split(' ')
    environments = []
    phonsets = []
    indexes = []
    maxindexes = []
    max = False
    # Creates a phoneme set for each element of the environment
    for ele in eles:
        phonsets.append(IPA.findSet(ele,IPA_info))
    # Creates index arrays to keep track of phoneme set indexes
    for phonset in phonsets:
        indexes.append(0)
        maxindexes.append(len(phonset) - 1)
    # Iterates through all possible combinations of syllables based on phoneme sets and order, using indexes array to
    #       keep track of position within the phoneme sets
    if -1 in maxindexes: # Returns empty array if any phonemeset is empty
        return []
    while not max: # Otherwise, generates all environments
        environ = ''
        for l in range(len(phonsets)):
            environ += phonsets[l][indexes[l]]
        environments.append(environ)
        indexes, max = helpers.increment(indexes, maxindexes) # Moves to next iteration
    return environments

# Parses a rule into the input phoneme, the output phoneme, and the environment
def parseRule(rule):
    rule = rule.split('>')
    inphon = rule[0]
    outphon, environment = rule[1].split('/')
    return inphon, outphon, environment

# Generates valid rules and ignores invalid rules
# Rules are slightly filtered for accuracy and efficiency
def validRule(ioe,indexes,phonemes):
    inphons, outphons, environments = ioe
    # Sets up validity variables
    validenviron = True
    validinphon = True
    validoutphon = True
    # Gets phonemes and environments
    inphon = inphons[indexes[0]]
    outphon = outphons[indexes[1]]
    environ = environments[indexes[2]]
    # Checks if inphon is in phonemes or is 0; cuts down on rules
    if inphon not in phonemes+'0':
        validinphon = False
    # Checks if outphon is inphon -- aka identity; ignores rule if it is, cuts down on rules
    if outphon == inphon:
        validoutphon = False
    # Checks each phoneme in environ
    for phon in environ:
        # If it's not in phonemes or if it isnt # or _, ignores rule; cuts down on rules
        if phon not in phonemes and phon not in '#_':
            validenviron = False
    rule = inphon + '>' + outphon + '/' + environ
    if validinphon and validoutphon and validenviron:
        return rule, True
    return rule, False

# Generates all phoneme rules based on feature rules
def generateRules(rule,phonemes,IPA_info):
    info_short = IPA_info[:4]
    rules = []
    # Checks to make sure rules are both valid and noncontradicting
    inphon, outphon, envi = parseRule(rule)
    environ = [item for item in envi.split(' ') if item != '_']
    ioe = [inphon,outphon]+environ
    ignoreRule = False
    for item in ioe:
        isvalid = helpers.validPhonemeSet(item, IPA_info)
        if isvalid == 2:
            print("Rule "+rule+" will be ignored.")
            ignoreRule = True
        elif isvalid == 1:
            print("Warning: Phoneme set "+item+" in rule "+rule+" contains features from both consonants and vowels."
                                                                    " This rule will be ignored.")
            ignoreRule = True
        elif helpers.checkConflict(item):
            print("Warning: Phoneme set "+item+" in rule "+rule+" contains contradicting features. This rule will "
                                                                    "be ignored.")
    if ignoreRule:
        return rules
    # Generates all rules with {X}:a replaced
    alpharules = replaceAlpha(rule,IPA_info)
    # Iterates through all generated rules
    for alpharule in alpharules:
        ignoreRule = False
        # Checks that the alpha generator didn't generate a rule with self-contradicting feature sets
        # Ignores the rule but doesn't alert the user if it did
        if helpers.checkConflict(alpharule):
            ignoreRule = True
        # Splits input, output, and environment
        inphonset, outphonset, environment = parseRule(alpharule)
        # Generates full set of input phonemes, output phonemes, and environments
        inphons = IPA.findSet(inphonset,info_short)
        outphons = IPA.findSet(outphonset,info_short)
        environments = generateEnvirons(environment,IPA_info)
        # Iterates through all parts of the rule
        ioe = [inphons, outphons, environments]
        maxindexes = [len(inphons)-1,len(outphons)-1,len(environments)-1]
        indexes = [0,0,0]
        max = False
        if -1 in maxindexes:
            max = True
        while not max:
            # Generates a rule and returns a value to define if the rule is valid
            validrule, valid = validRule(ioe,indexes,phonemes)
            if valid and not ignoreRule: # If the rule is valid...
                rules.append(validrule) # Add the rule
            # Move on to next potential rule
            indexes, max = helpers.increment(indexes,maxindexes)
    return rules


# Changes the sounds in a single word
def changeWord(word, rule):
    # Adds word boundaries
    word = '#'+word+'#'
    # Sets up input phoneme, output phoneme, and environment
    inphon, outphon, environ = parseRule(rule)
    # If epenthesis, insert 0 between every character in word and environment to search for epenthesis context
    if inphon == '0':
        newword = '0'
        for letter in word:
            newword += letter+'0'
        word = newword
        newenviron = '0'
        for l in range(len(environ)-1):
            if environ[l] != '_' and environ[l+1] != '_':
                newenviron += environ[l]+'0'
            else:
                newenviron += environ[l]
        newenviron += environ[-1]
        if environ[-1] != '_':
            newenviron += '0'
        environ = newenviron
    # Creates a substring that is the environment to find in and a replacement substring
    findenviron = replaceSubstring(environ,'_',inphon)
    replacement = replaceSubstring(environ,'_',outphon)
    # Replaces environment with replacement environment
    while findSubstring(word,findenviron) != -1:
        word = replaceSubstring(word,findenviron,replacement)
    # Removes 0 and # (deletion and end of word symbols)
    while findSubstring(word,'0') != -1:
        word = replaceSubstring(word,'0','')
    while findSubstring(word,'#') != -1:
        word = replaceSubstring(word,'#','')
    return word

# Changes all the words with a given sound
def changeSound(words, rule):
    newWords = []
    for word in words:
        newWords.append(changeWord(word,rule))
    return newWords


# Implements all sound changes
def allChanges(dir):
    filelist = os.listdir(dir+'/inputs')
    IPA_info = IPA.readInIPA(dir)
    # Counts up number of "change" files in input directory, aka how many stages of sound change
    numchanges = 0
    for file in filelist:
        if "changes" in file:
            numchanges += 1
    # Goes through each soundchange file
    # i is important because it keeps track of which sound change and words file we're on
    if numchanges == 0:
        print("No soundchange files found")
        return
    for i in range(numchanges):
        # Gets soundchanges from file as a list
        soundchanges = [line.strip() for line in codecs.open(dir+"/inputs/changes"+str(i+1)+".txt",'r','utf8')] # Sound change file
        # Gets all the words from the previous word file (starting at 0)
        try: # Works if words#.txt exists
            words = [line.strip() for line in codecs.open(dir+"/outputs/words"+str(i)+".txt",'r','utf8')] # existing words file
        except IOError: # But if not, gives error statement and returns nothing
            print("No words0 file found")
            return
        # Generates list of phonemes
        # Used for rule creation (to limit number of rules made to save time by not running through every possible rule)
        phonemes = ''.join(list(set(''.join(words)))) # Sets up phonemes from existing words
        outfile = open(dir+"/outputs/words"+str(i+1)+".txt",'w') # output file
        for soundchange in soundchanges:
            newchange = generateRules(soundchange,phonemes,IPA_info)
            for change in newchange:
                #print(change)
                words = changeSound(words, change)
        outfile.write('\n'.join(words)) # puts changed words into the output file for use in the next change iteration
        outfile.close()