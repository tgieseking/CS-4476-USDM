import os
import numpy as np
from vocabulary import createVocab, createHist

all_sifts = np.load('Caltech_all_sifts_20img_20feat.npy')	
vocab_size = 2000
print "Kmeans start"
vocab = createVocab(all_sifts, vocab_size)
print "Kmeans done"
first_time = True
parent_dir = os.path.abspath(os.path.dirname(__file__))					# prints current directory path
os.chdir('Caltech_SIFTs')
hists = np.array([])
for label in os.walk('.').next()[1]:
	os.chdir(os.path.join(os.getcwd(), label))
	print os.getcwd()
	sift_names = os.walk('.').next()[2]	
	for name in sift_names:
		sift_vec = np.load(name)
		if len(sift_vec) > 1:
			if first_time:
				hists = createHist(sift_vec, vocab, vocab_size)
				first_time = False
			else:
				hists = np.vstack((hists, createHist(sift_vec, vocab, vocab_size)))
	os.chdir('..')

os.chdir('..')

np.save('Caltech_sift_hists_stdKmeans_20img_20feat_Minibatch.npy', hists)