# Language Generation Model
Overview:

This is half of a final project for a Diachronic Linguistics (historical linguistics) course. The purpose of this half
of the project is to design a program that can generate a "toy language" given sounds and sound rules and evolutionary
stages given sound change rules. It can be used to model real languages and their descendants or constructed languages.
All of the parts of a language constructed by this program are organized into a single directory with that language's
name (e.g. "L1", "L2", etc.)

This program is made to be highly modifiable and capable of adjusting to many different forms of input. This means that
the IPA directory can be edited as needed to accommodate whatever language is being developed. However, the key feature
is that the only thing that needs to be edited are the text files (the IPA{C/V}.txt, IPA{C/V}KEY.txt, and language
directory files). The python files should need no modification (ideally). This makes this project useful for anyone who
does not wish to spend excessive amounts of time programming but wants to utilize what this program has to offer.

README TOC:
- Quick guide
- IPA directory
- Contents of language directories
  - Inputs directory
  - Outputs directory
- The program
  - Program usage instructions

------------------------------------------------------------------------------------------------------------------------
## Quick guide 

If you wish to start using this program right away but don't care about the details of how it works, start with this
section.

The three things you need to utilize this program are:
- A set of phonemes
- A set of syllable rules
- One or more sets of sound changes (optional)

#### Step 1.

Create a directory *MyLang* with two subdirectories, *inputs* and *outputs*

#### Step 2.

Create a file in the inputs directory named phonemes.txt. Enter your phonemes into this file, each separated by either a
space or a newline, and save.

#### Step 3.

Create a file in the inputs directory named sylstructs.txt. Enter your syllable rules, and format them such that the 
onset, nucleus, and coda are separated by the | symbol, and each subset of the onset, nucleus, and coda are sparated by 
the ; symbol. For an explanation on how to use sets of phonemes, see the IPA directory section.

#### Step 4.

Create any number of changes#.txt files, where # starts at 1 and goes up by 1 each time. Fill these in with synchronic
or diachronic sound change rules (see IPA directory again for explanation on sets). Sound changes will be applied from 
first to last.

#### Step 5.

Run the program. In the terminal, navigate to the main directory, and enter the following command.

`python createlang.py swc <my-language-directory> <max-syllable-count> <number-of-words>`

This will generate all the syllables possible in the language in `<my-language-directory>`, a number of words equal to 
`<numer-of-words>`, with words going up to length `<max-syllable-count>`, and all their evolutionary stages as described
in your changes files. If you are opting not to have sound changes, you can remove the `c` from `swc`

------------------------------------------------------------------------------------------------------------------------
## IPA directory                                                  

*** NOTE ***

The IPA directory is now part of the language directories, so that you can have separate inventories for separate
languages. However, this part of the documentation will remain separate as it is a larger section to discuss.

The files:
- IPAC.txt -- The consonants IPA chart
- IPAV.txt -- The vowels IPA chart
- IPACKEY.txt -- The lookup keys for IPAC.txt
- IPAVKEY.txt -- The lookup keys for IPAV.txt

These files can be modified to add, remove, or rearrange sounds as needed/desired.


-- Format of IPAC.txt --

                                 0   1   2   3   4   5   6   7   8   9  10
                            0   b,p 0,0 0,0 d,t 0,0 ɖ,ʈ ɟ,c g,k ɢ,q 0,0 0,ʔ
                            1   β,ɸ v,f ð,θ z,s ʒ,ʃ ʐ,ʂ ʝ,ç ɣ,x ʁ,χ ʕ,ħ ɦ,h
                            2   0,0 0,0 0,0 ɮ,ɬ 0,0 0,0 0,0 0,0 0,0 0,0 0,0
                            3   m,0 ɱ,0 0,0 n,0 0,0 ɳ,0 ɲ,0 ŋ,0 ɴ,0 0,0 0,0
                            4   w,0 ʋ,0 0,0 ɹ,0 0,0 ɻ,0 j,0 ɰ,0 0,0 0,0 0,0
                            5   0,0 0,0 0,0 l,0 0,0 ɭ,0 ʎ,0 ʟ,0 0,0 0,0 0,0
                            6   ʙ,0 0,0 0,0 r,0 0,0 0,0 0,0 0,0 ʀ,0 0,0 0,0
                            7   0,0 ⱱ,0 0,0 ɾ,0 0,0 ɽ,0 0,0 0,0 0,0 0,0 0,0

This is the standard IPA consonants chart (although the voiced and voiceless consonants are flipped). 0 represents
a sound not found in the IPA.
(The numbered rows and columns are not present in the file, they are solely so that the IPACKEY.txt format is explained
better)


-- Format of IPACKEY.txt --

       ATTRIBUTE range,range;dimension
where
    ATTRIBUTE = 4-letter abbreviation for attribute name
    range,range = bottom to top of index range
    dimension = place/manner/voicing dimension for range (0 = place, 1 = manner, 2 = voicing)

Since IPAC is read in as a 3 dimensional array, the way sets of sounds are extracted is by finding the ranges in which
they reside. To find a singular sound, you can input three attributes: place, manner, and voicing. You are more likely,
however, to use this to find sets of sounds that fall under certain categories, such as voiced stops or alveolar
fricatives.
This was designed to directly mimic how phonological feature systems work. It was implemented with phonological rules
and syllable rules in mind, specifically for constructing syllables and implementing sound changes (see below).

Some examples (see the IPAC table for reference to see how this works)

- CONS 0,0;0 -- All consonants (this is special, the IPA reading file will interpret this value as instructions to fill in
                the entire chart)
- NASA 3,4;0 -- Nasals
- CONT 1,5;0 -- Continuants
- DENT 2,3;1 -- Dentals
- DORS 5,11;1 -- Dorsals
- VLSS 1,2;2 -- Voiceless


-- Format of IPAV.txt --

                                                 1   2   3
                                            1   i,y ɨ,ʉ ɯ,u
                                            2   ɪ,ʏ 0,0 0,ʊ
                                            3   e,ø ɘ,ɵ ɤ,o
                                            4   0,0 ə,0 0,0
                                            5   ɛ,œ ɜ,ɞ ʌ,ɔ
                                            6   æ,0 ɐ,0 0,0
                                            7   a,ɶ 0,0 ɑ,ɒ

This is the standard IPA chart for vowels. Note that there are 7 height levels instead of 4: 2 and 6 represent
near-closed and near-open, respectively, since they do not have their own formal tiers in the IPA; 4 represents mid,
which also does not have its own tier. Again, 0 represents sounds not found in the IPA.
(The numbered rows and columns are not present in the file, they are solely so that the IPAVKEY.txt format is explained
better)


-- Format of IPAVKEY.txt --

      ATTRIBUTE range,range;dimension
where
    ATTRIBUTE = 4-letter abbreviation for attribute name
    range,range = bottom to top of index range
    dimension = height/backness/rounding dimension for range (0 = height, 1 = backness, 2 = rounding)

Since IPAV is read in as a 3 dimensional array, the way sets of sounds are extracted is by finding the ranges in which
they reside. To find a singular sound, you can input three attributes: height, backness, and rounding. You are more
likely, however, to use this to find sets of sounds that fall under certain categories, such as front high vowels or
back round vowels.
This was designed to directly mimic how phonological feature systems work. It was implemented with phonological rules
and syllable rules in mind, specifically for constructing syllables and implementing sound changes (see below).

Some examples (see the IPAV table for reference to see how this works)

- VOWL 0,0;0 -- All vowels (see: all consonants in previous section for explanation)
- MIDC 2,3;0 -- Mid-closed vowels
- MIDV 3,4;0 -- Mid-vowels (schwa)
- CENT 1,2;1 -- Central vowels
- ROND 1,2;2 -- Round vowels

*** NOTES ***

- Digraphs are accounted for, and encouraged if they help, but it is recommended that you
only use digraph symbols for one concept and you do not reuse symbols found elsewhere in the IPA (e.g. gW, kW, dW for
labialized stops, aN, oN, uN for nasalized vowels).
- It is recommended that when editing the charts, you minimize the work you have to do for yourself while still making
it functional for your own purposes. For example, if modeling French vowels, it would be recommended to add a third and
fourth value to the roundness dimension (2). This way, you can have ɛ,œ,ɛN,œN where N means nasalized, and you do the
same for all other nasal vowels. If you do this, it is easy to add NASL 2,4;2 (nasalized), URNA 2,3;2 (unround nasal),
and RONA 3,4;2 (rounded nasal) to the IPAVKEY file. This minimizes the work you have to do adjusting the IPAVKEY file.
If you only have a few sounds to add, such as labialized stops, it may be easier or make more sense to simply add a new
row or column for them. Again, if you do this, be sure to adjust the KEY files.

------------------------------------------------------------------------------------------------------------------------
## Contents of a language's directory

The inputs directory:
- changes{n}.txt -- The diachronic sound changes for the language from stage n-1 to stage n
- phonemes.txt -- All of the phonemes found in the language at stage 0
- sylstructs.txt -- All of the syllable structure rules found in the language at stage 0

The outputs directory:
- syllables.txt -- All the possible syllables for the language at stage 0
- words0.txt -- n randomly generated words with up to m syllables per word, where n and m are user-defined
- words{n}.txt -- All of the words from the file words0.txt after undergoing changes from the changes{n}.txt file

#### Input file formats

-- changes{n}.txt --

This text file takes the form of classic rewrite rules found in synchronic and diachronic phonology.
The following is the general format:

    i>o/c1 _ c2

where
```
i = input phoneme
> = becomes
o = output phoneme
/ = in the context of
c1 = context 1
_ = position of i
c2 = context 2
```

Further detail:
i, o, c1, and c2 can take the form of individual phonemes (or sequences of phonemes in the case of c1 and c2), or they
can take the form of phoneme set descriptors. These phoneme set descriptors can be modified in the IPACKEY.txt and
IPAVKEY.txt files. See the IPA section above for explanation of how these descriptors work.

```
w>0/_ +CLOS,+BACK
    w deletes before closed back vowels
s>θ/# _
    s becomes θ word initially
a>æ/+CONS _ +CONS
    a becomes æ between consonants
*+NRCL,BACKNESS:a,ROUNDING:a>+CLOS,BACKNESS:a,ROUNDING:a/_
    *Near closed back vowels become their closed counterparts everywhere
*+LATA>+LATA,PLACE:a/_ +APPR,PLACE:a
    *Lateral approximants assimilate to place of articulation before approximants
```

*** NOTE ABOUT {X}:a ***

All rules above marked with * contain at least two phoneme set descriptors with an attribute {X}:a.
This notation has one function that is used in two ways. It is based on α-notation (alpha-notation).

Function: This notation tells the program to create multiple rules based on one rule, replacing the attribute with
all its possible realizations (PLACE>BILA,LABD,DENT,etc;ROUNDING>ROND,URND;etc) -- but specifically, it will always
replace {X}:a with the same realization. In other words, {X}:a will generate a set of parallel rules. Example:
PLACE:a>+STOP,PLACE:a becomes +BILA>+STOP,+BILA; +ALVE>+STOP,+ALVE; etc.

Use 1: This allows for us to make sure all sound that change in one attribute to retain their other attributes. This
is because if we input a traditional rule STOP>VOICED/_VOICED, where t would become d before b, g, etc., unless we
use this specific strategy, t will become _all_ voiced stops before voiced consonants. In other words, t will become
b and g as well as d. The {X}:a format requires all stops tht become voiced to retain their other qualities. Thus,
t will only become d.

Use 2: We can also use it for the traditional usage of alpha. In this case, we can make the output phoneme mirror
the context phonemes in some aspect.


-- phonemes.txt --

This textfile contains all the phonemes used for stage 0 of the language.
This is the format:

                                        i ɪ e ɛ u ʊ o ɔ a
                                        p b f d t s n ɹ l g k

The file can have any number of lines for organization's sake, so long as all the phonemes are separated by either a
single space or a newline.


-- sylstructs.txt --

This textfile contains the rules required for syllable structures.
This is the format:

```
+CONS;+CONS,-APPR +APPR;p f|+VOWL|;+CONS;+APPR +CONS,-APPR;f p;
```

This file is split on bars to separate the onsets (before the first bar), nuclei (between bars), and codas (after
second bar). Then, each syllable segment is split on semicolons. This allows the generator to create all the possible
syllable combinations.

#### Output file formats

-- syllables.txt --

This file contains all possible syllables for language stage 0 given the phonemes and syllable structures. It is
created in order to generate words for words0.txt.


-- words0.txt --

This file contains a user-set number of words of up to a user-set number of syllables (the amount of words per syllable
level is weighted and pre-defined, see below).


-- words{n}.txt --

This file contains the words from words{n-1}.txt after having undergone the changes from changes{n}.txt



------------------------------------------------------------------------------------------------------------------------
## The program

The files:
- createlang.py
- helpers.py
- IPA.py
- soundchange.py
- sylconstr.py
- wordconstr.py


#### createlang.py

This is the main program file. This program takes in 2-4 arguments:
- `swc` -- syllables, words, changes. This argument, which can consist of the letters 's', 'w', and/or 'c', tells the 
program what to construct. If s, it constructs the syllables. If w, it constructs words. If c, it implements changes.
- `dir` -- the directory of the language to be created or modified.
- `<max syls>` -- maximum number of syllables in a word. (Only used with c argument)
- `<words>` -- number of words to create. (Only used with c argument)

The program itself functions by going through the main functions of soundchange.py, sylconstr.py, and wordconstr.py.


#### helpers.py

This program contains three helpers

- increment(indexes,maxindexes):

This function serves as part of a loop algorithm to replace loops that are several layers deep.

This function is used in various parts of the program. It was designed to help implement arbitrary-depth `for` loops.
If we have something like an arbitrary-length syllable where each character position is a set of phonemes and we want
all possible combinations of the sets of phonemes, we could implement a series of nested for loops, but that would
be messy, hard to follow, and couldn't handle variable numbers of elements.

So, this solution is to make a 2D array where each each subarray is all the possible elements for one position in
the syllable, and then create a second array, maxindexes, where each element is the length of the array in the
corresponding index in the 2D array. We then create a third array, indexes, that is a list of integers the same
length as maxindexes and initialize them at 0. We then use each integer in the indexes array to keep track of which
element of the main array we are on, and we use this following helper function to increment the indexes array in a
pattern much like counting up in the 1's place until you hit 10, then resetting the 1's place to 0 and incrementing
the 10's place. We repeat this pattern until the indexes array is the same as the maxindexes array.

Using these indexes to access elements of the main 2D array, we can iterate through all combinations of the 2D array

- validPhonemeSet(features,IPA_info):

This function checks to make sure a phoneme set doesn't contain descriptors from both the consonant and vowel chart.

- checkConflict(features):

This function checks to make sure no two descriptors in a feature description contradict each other (e.g. -CLOS,+CLOS)


#### IPA.py

This file reads in the IPA directory and processes the text files to be used in other parts of the program.

#### soundchange.py

This file implements the sound changes in each sound change file. It does so by taking each rule in the change#.txt
files and generating all possible phoneme combinations for that rule, then implementing the specific rules on each word
in the words#.txt files.

#### sylconstr.py

This file constructs all possible syllables given phoneme and sylstruct files in a language's input directory. It does
so by taking all the phonemes in the phonemes.txt file and all the syllable rules in the sylstructs.txt file and
generating the possible phoneme combinations.

#### wordconstr.py

This file constructs the input number of words given the syllables generated by sylconstr.py. A simpler program, it
just generates a number of randomized words based on the syllables previously generated.