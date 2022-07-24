'''                                             PERCEPTRON!!!                                                        '''
'''
    Tried on desktop, but i think I got it better this time.
'''
import numpy as np
from random import sample, shuffle
from math import e


class Perceptron:

    def __init__(self,datadir,langs,A="bin",E="bin"):
        self.langs = {langs[l]:l for l in range(len(langs))}
        self.keys = {l:langs[l] for l in range(len(langs))}
        self.trainset = self._getSentences(datadir,self.langs,"train")
        self.testset = self._getSentences(datadir,self.langs,"test")
        self.allfeats = self._genFeats(' '.join([l[1] for l in self.trainset]))
        self.E = E
        self.A = A
        self.w = np.random.rand(len(self.allfeats))
        if E == "bin":
            self.w = np.zeros(len(self.allfeats),dtype=int)

    def _getSentences(self,datadir,langs,mode):
        combined = []
        for lang in langs:
            lines = [line.strip() for line in open(datadir+lang+mode+".txt") if line != '']
            for line in lines:
                for l in line.split('. '):
                    if l != '':
                        combined.append((langs[lang],l))
        return combined

    def _genFeats(self,text):
        bigrams = list(zip(text,text[1:]))
        feats = dict()
        b = 0
        for bg in bigrams:
            if bg not in feats:
                feats[bg] = b
                b += 1
        return feats

    def _encode(self,sent):
        bigrams = list(zip(sent,sent[1:]))
        vect = np.zeros(len(self.allfeats),dtype=int)
        for bg in bigrams:
            if bg in self.allfeats:
                vect[self.allfeats[bg]] = 1
        return vect

    def _A(self,ttl):
        if self.A == "bin":
            if ttl > 0:
                return 0
            return 1
        elif self.A == "sig":
            return 1 / (1 + e**(-ttl))

    def _E(self,d,y):
        if self.E == "bin":
            return d - y
        elif self.E == "MSE":
            return (1/len(self.w))*(d-y)**2

    def _dA(self,out):
        if self.A == "bin":
            return 1
        elif self.A == "sig":
            return out * (1 - out)

    def _dE(self,d,w,x,err,out):
        if self.E == "bin":
            return x*w
        elif self.E == "MSE":
            n = len(self.w)
            xw = self._A(w*x)
            derived = self._dA(out)
            change = (-(x) / n) * (err) * derived
            if x > 0:
                print("\nout",out)
                print("x:",x)
                print("err:",err)
                print("derived:",derived)
                print("change:",change)
            return change

    def _forward(self,vect):
        out = self._A(np.dot(vect,self.w))
        return out

    def _backward(self,d,vect,out):
        err = self._E(d,out)
        for w in range(len(self.w)):
            self.w[w] += self._dE(d,self.w[w],vect[w],err,out)

    def train(self,iters):
        for i in range(iters):
            print("\niter:",i)
            d, s = sample(self.trainset,1)[0]
            vect = self._encode(s)
            out = self._forward(vect)
            self._backward(d,vect,out)
            print(out)

    def test(self,num):
        shuffle(self.testset)
        for i in range(num):
            d, s = self.testset[i]
            vect = self._encode(s)
            y = self._forward(vect)

    def testall(self):
        c = 0
        ttl = len(self.testset)
        for item in self.testset:
            d, s = item
            vect = self._encode(s)
            y = self._forward(vect)
            if round(y) == d:
                c += 1
        print(f"correct: {c}, total: {ttl}, accuracy: {c/ttl}")






if __name__ == "__main__":
    datadir = "langdata/"
    langs = ["en","de"]
    P = Perceptron(datadir,langs,A="sig",E="MSE")
    '''print(P.trainset)
    print(len(P.trainset))
    print(P.allfeats)'''
    '''encoded = P._encode(P.testset[0][1])
    print(encoded)'''
    #print(P.w[:10])
    P.train(20)
    #P.test(5)
    P.testall()
    #print(P.w[:10])