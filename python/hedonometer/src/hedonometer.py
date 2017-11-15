import sys, os
from labMTsimple.storyLab import *
import codecs
import glob

dates = glob.glob('../data/*')
outfile = open('date_happiness.csv', 'wb')

if __name__ == '__main__':
  labMT,labMTvector,labMTwordList = emotionFileReader(stopval=0.0, lang='english', returnVector=True)
  
  for date in dates:
    f = codecs.open(date,"r","utf8")
    infile = f.read()
    f.close()
    date = date.split('/')[1]

    ## compute valence score
    print 'computing happiness on ' + str(date)  
    Valence, Fvec = emotion(infile, labMT, shift=True, happsList=labMTvector)
    StoppedVec = stopper(Fvec, labMTvector, labMTwordList, stopVal=1.0)
    Valence = emotionV(StoppedVec, labMTvector)
    print 'The valence of {0} is {1:.5}'.format(date,Valence)
    print
    outfile.write(str(date).replace('.txt','')+','+str(Valence)+'\n')
outfile.close()
