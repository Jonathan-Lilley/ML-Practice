'''                     HELPER FUNCTIONS                    '''

# Increments all elements in an integer array and returns the array and a bool indicating if the numbers are at max
# This function is used in various parts of the program. It was designed to help implement arbitrary-depth for loops.
# If we have something like an arbitrary-length syllable where each character position is a set of phonemes and we want
#   all possible combinations of the sets of phonemes, we could implement a series of nested for loops, but that would
#   be messy, hard to follow, and couldn't handle variable numbers of elements.
#   So, this solution is to make a 2D array where each each subarray is all the possible elements for one position in
#   the syllable, and then create a second array, maxindexes, where each element is the length of the array in the
#   corresponding index in the 2D array. We then create a third array, indexes, that is a list of integers the same
#   length as maxindexes and initialize them at 0. We then use each integer in the indexes array to keep track of which
#   element of the main array we are on, and we use this following helper function to increment the indexes array in a
#   pattern much like counting up in the 1's place until you hit 10, then resetting the 1's place to 0 and incrementing
#   the 10's place. We repeat this pattern until the indexes array is the same as the maxindexes array.
#   Using these indexes to access elements of the main 2D array, we can iterate through all combinations of the 2D array
# The overall algorithm in pseudocode:
#   1. combine lists 1-n into one list, allLists
#   2. initialize indexes array of length n to 0
#   3. initialize maxindexes array of length n such that maxindexes[i] = len(allLists[i])
#   4. while indexes != maxindexes:
#       a. create item s.t. item = allLists[0][indexes[0]]+allLists[1][indexes[1]]+...+allLists[n][indexes[n]]
#           e.g. if allLists[0] = [p,d,c,e], allLists[1] = [a,f,g,k], and indexes = [2,3], then
#               item = allLists[0][2]+allLists[1][3], or ck
#       b. increment(maxindexes)
# The algorithm of increment in pseudocode:
#   1. take in indexes and maxindexes
#   2. for each position in indexes:
#       a. if that position in indexes is at max, aka if indexes[position] == maxindexes[position], set it to 0
#       b. otherwise, increment it by one and break out of the loop
#   3. check if each index is 0; if not, it is not at max; if so, it is at max
#   4. return the new indexes array and whether or not it is maxed out
def increment(indexes,maxindexes):
    max = True
    for i in range(len(indexes)):
        if indexes[i] == maxindexes[i]:
            indexes[i] = 0
        else:
            indexes[i] += 1
            break
    for i in indexes:
        if i != 0:
            max = False
    return indexes, max

# Checks to see if there are phoneme descriptors from different charts (invalid set)
# Return values:
#   0: valid set
#   1: invalid set
#   2: invalid feature in set
def validPhonemeSet(features,IPA_info):
    IPAC, IPAV, IPACKEY, IPAVKEY = IPA_info[:4]
    feats = features.split(',')
    valid = 0 # Keeps track of if a featureset is valid, default is 0 (valid)
    # Checks which chart the first feature is in
    if len(feats[0]) <= 2:
        return valid
    if feats[0][1:] in IPACKEY or feats[0] in ["PLACE:a","MANNER:a","VOICING:a"]:
        featset = 'c'
    elif feats[0][1:] in IPAVKEY or feats[0] in ["HEIGHT:a","BACKNESS:a","ROUNDING:a"]:
        featset = 'v'
    else: # If something goes wrong and it's not in either, follow special actions
        print("Warning: Descriptor "+feats[0][1:]+" not in IPACKEY or IPAVKEY.")
        featset = 'x'
        valid = 2
    # Go through the rest of the features and check which chart they're in
    for feat in feats:
        if feat[1:] not in IPAVKEY and feat[1:] not in IPACKEY and feat not in ["PLACE:a","MANNER:a","VOICING:a",
                                                                                "HEIGHT:a","BACKNESS:a","ROUNDING:a"]:
            print("Warning: Descriptor "+feat+" not in IPACKEY or IPAVKEY.")
            featset = 'x'
            valid = 2
        elif featset == 'c' and feat[1:] in IPAVKEY:
            valid = 1
        elif featset == 'v' and feat[1:] in IPACKEY:
            valid = 1
    return valid

# Checks to see if there are conflicting features in a single phoneme set description
def checkConflict(features):
    feats = features.split(',')
    # Goes through all features
    if len(feats[0]) <= 2:
        return False
    for f in range(len(feats)):
        # If the opposite of a feature exists in the rest of the set, return true (conflicting features)
        if feats[f][0] == '-' and '+'+feats[f][1:] in feats[f+1:]:
            return True
        if feats[f][0] == '+' and '-'+feats[f][1:] in feats[f+1:]:
            return True
    # Otherwise return false (nonconflicting features)
    return False
