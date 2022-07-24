'''                 WORD CONSTRUCTOR                    '''

import random


# Generates a given number of words given syllables and number of syllables/word
def gen_words(syls, sylnum, numwords):
    # Sets up a set for the words (set for faster lookup)
    words = set()
    # Sets up a maximum number of words possible,
    # based on possible combination count of syllables
    max = len(syls)**sylnum
    for i in range(numwords):
        # Breaks if all possible words have been generated
        if i == max:
            break
        # Create word and keep recreating
        # as long as the word does not exist in the set
        word = ''.join([random.choice(syls) for syl in range(sylnum)])
        while word in words:
            word = ''.join([random.choice(syls) for syl in range(sylnum)])
        # add word to set
        words.add(word)
    return words


# Makes words given a number of syllables and a number of words to make
def make_words(dir, numSyls, numwords):
    wordfile = open(dir+"/outputs/words0.txt",'w')
    words = []
    try:
        syls = [line.strip() for line in open(dir+"/outputs/syllables.txt")]
    except IOError:
        print("No syllables file")
        wordfile.write('')
        wordfile.close()
        return
    if len(syls) == 0:
        print("No syllables in syllables file")
        wordfile.write('')
        wordfile.close()
        return
    # Defines syllable level weights
    if numSyls == 1:
        sylstats = [1]
    elif numSyls == 2:
        sylstats = [0.22, 0.78]
    elif numSyls == 3:
        sylstats = [0.15, 0.51, 0.34]
    else:
        sylstats = [1/numSyls for i in range(numSyls)]
    # Sets up exact wordcount for each syllable level
    # based on syllable level weights and total word count
    wordnums = sylstats
    for i in range(len(sylstats)):
        wordnums[i] = int(numwords//(1/sylstats[i]))
        print(wordnums[i])
    if sum(wordnums) != numwords: # Adjusts to ensure word count is met
        wordnums[-1] = numwords-sum(wordnums[:-1])
    # Generate words for each syllable level
    for sylnum in range(numSyls):
        words += gen_words(syls, sylnum+1, wordnums[sylnum])
    wordfile.write('\n'.join(words))
    wordfile.close()