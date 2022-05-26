"""
Language Modeling Project
Name:
Roll No:
"""

import language_tests as test

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):
    file = open(filename, "r")
    lst = []
    for i in file:
        if len(i) > 1:
            i = i.strip()
            lst1 = i.split()
            lst.append(lst1)
    file.close()
    return lst


'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):
    count = 0
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            count +=1
    return count


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):
    vocab = []
    for word in corpus:
        for unique in word:
            if unique not in vocab:
                vocab.append(unique)
    return vocab



'''
countUnigrams(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):
    unigram = {}
    for word in corpus:
        for unique in word:
            if unique not in unigram:
                unigram[unique] =0
            unigram[unique] += 1
    return unigram


'''
getStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def getStartWords(corpus):
    vocab = []
    for i in range(len(corpus)):
        if corpus[i][0] not in vocab:
            vocab.append(corpus[i][0])
    return vocab

    
'''
countStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countStartWords(corpus):
    l = {}
    for i in range(len(corpus)):
        word = corpus[i][0]
        if word not in l:
            l[word] = 1
        else:
            l[word] += 1
    return l


'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    bigram_dict = {}
    for i in range(len(corpus)):
        for j in range(len(corpus[i])-1):
            if corpus[i][j] not in bigram_dict:
                bigram_dict[corpus[i][j]] = {}
                bigram_dict[corpus[i][j]][corpus[i][j+1]] = 1
            else:
                if corpus[i][j+1] in bigram_dict[corpus[i][j]]:
                    bigram_dict[corpus[i][j]][corpus[i][j+1]] += 1
                else:
                    bigram_dict[corpus[i][j]][corpus[i][j+1]] = 1
    return bigram_dict


### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    list = []
    for i in unigrams:
        list.append(1/len(unigrams))
    return list


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    list = []
    for i in unigrams:
        list.append(unigramCounts[i]/totalCount)
    return list


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    bigram = {}
    for i in bigramCounts:
        words = []
        probs = []
        temp = {}
        for j in bigramCounts[i]:
            words.append(j)
            probs.append(bigramCounts[i][j]/unigramCounts[i])
            temp["words"] = words
            temp["probs"] = probs
        bigram[i] = temp
    return bigram


'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):
    word = {}
    for i in range(len(probs)):
        if(words[i] not in ignoreList):
            word[words[i]] = probs[i]
    topWords = {}
    while(len(topWords) < count):
        maximum = 0
        for i in word:
            if i not in topWords:
                if word[i]>maximum:
                    maximum = word[i]
                    keys = i
        topWords[keys] = maximum
    return topWords


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices
def generateTextFromUnigrams(count, words, probs):
    s = " "
    for i in range(count):
        l = choices(words, weights=probs)
        s = s + " " + l[0]
    return s
    


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    words = []
    for i in range(count):
        if (len(words) == 0 or words[-1] == "."):
            word = choices(startWords, startWordProbs)
            words.append(word[0])
        else:
            last_word = words[-1]
            word_prob_dict = bigramProbs[last_word]
            word = choices(word_prob_dict['words'], word_prob_dict['probs'])
            words.append(word[0])
    sentence = " "
    return (sentence.join(words))


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    count = getCorpusLength(corpus)
    vocab = buildVocabulary(corpus)
    unigram = countUnigrams(corpus)
    prob = buildUnigramProbs(vocab, unigram, count)
    bar = getTopWords(50, vocab, prob, ignore)
    barPlot(bar,"Top 50 words")
    return


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    start_words = getStartWords(corpus)
    count = countStartWords(corpus)
    value = count.values()
    value = sum(value)
    prob = buildUnigramProbs(start_words, count, value) 
    bar = getTopWords(50, start_words, prob, ignore) 
    barPlot(bar,"Top 50 Start words")
    return


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    unigrams = countUnigrams(corpus)
    bigrams = countBigrams(corpus)
    words_probability = buildBigramProbs(unigrams, bigrams)
    dictionary = words_probability[word]
    words = dictionary['words']
    probs = dictionary['probs']
    bar = getTopWords(10, words, probs, ignore)
    barPlot(bar, "Top 10 Occurences of a Word")
    return 
    


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):
    total_count1 = getCorpusLength(corpus1)
    unigrams1 = buildVocabulary(corpus1)
    unigram_count1 = countUnigrams(corpus1)
    probability1 = buildUnigramProbs(unigrams1, unigram_count1, total_count1)
    dictionary1 = getTopWords(topWordCount, unigrams1, probability1, ignore)
    total_count2 = getCorpusLength(corpus2)
    unigrams2 = buildVocabulary(corpus2)
    unigram_count2 = countUnigrams(corpus2)
    probability2 = buildUnigramProbs(unigrams2, unigram_count2, total_count2)
    dictionary2 = getTopWords(topWordCount, unigrams2, probability2, ignore)
    dictionary = {}
    topWords = []
    for i in dictionary1:
        topWords.append(i)
    for i in dictionary2:
        if i not in topWords:
            topWords.append(i)
    corpus1Probs = []
    for i in topWords:
        if i in dictionary1:
            corpus1Probs.append(dictionary1[i])
        else:
            corpus1Probs.append(0)
    corpus2Probs = []
    for i in range(len(topWords)):
        if topWords[i] in dictionary2:
            corpus2Probs.append(dictionary2[topWords[i]])
        else:
            corpus2Probs.append(0)
    dictionary['topWords'] = topWords
    dictionary['corpus1Probs'] = corpus1Probs
    dictionary['corpus2Probs'] = corpus2Probs
    return dictionary


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    result_dict = setupChartData(corpus1, corpus2, numWords)
    sideBySideBarPlots(result_dict['topWords'], result_dict['corpus1Probs'], result_dict['corpus2Probs'], name1, name2, title)
    return


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    result_dict = setupChartData(corpus1, corpus2, numWords)
    scatterPlot(result_dict['corpus1Probs'], result_dict['corpus2Probs'], result_dict['topWords'], title)
    return


### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()

    ## Uncomment these for Week 2 ##

    # print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    # test.week2Tests()
    # print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek2()


    ## Uncomment these for Week 3 ##

    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
