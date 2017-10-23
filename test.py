import math
import numpy

def getRefDict(words, ngram):
	lens = len(words)
	now_ref_dict = {}
	for n in range(1, ngram + 1):
		for start in range(lens - n + 1):
			gram = ' '.join([str(p) for p in words[start : start + n]])
			if gram not in now_ref_dict:
				now_ref_dict[gram] = 1
			else:
				now_ref_dict[gram] += 1
	return now_ref_dict, lens

def my_log(a):
	if a == 0:
		return -1000000
	return math.log(a)

def calBleu(x, ref_dict, lens, ngram):
	length_trans = len(x)
	words = x
	closet_length = lens
	sent_dict = {}
	for n in range(1, ngram + 1):
		for start in range(length_trans - n + 1):
			gram = ' '.join([str(p) for p in words[start : start + n]])
			if gram not in sent_dict:
				sent_dict[gram] = 1
			else:
				sent_dict[gram] += 1
	correct_gram = [0] * ngram
	for gram in sent_dict:
		if gram in ref_dict:
			n = len(gram.split(' '))
			correct_gram[n - 1] += min(ref_dict[gram], sent_dict[gram])
	bleu = [0.] * ngram
	smooth = 0
	for j in range(ngram):
		if correct_gram[j] == 0:
			smooth = 1
	for j in range(ngram):
		if length_trans > j:
			bleu[j] = 1. * (correct_gram[j] + smooth) / (length_trans - j + smooth)
		else:
			bleu[j] = 1
	brev_penalty = 1
	if length_trans < closet_length:
		brev_penalty = math.exp(1 - closet_length * 1. / length_trans)
	logsum = 0
	for j in range(ngram):
		logsum += my_log(bleu[j])
	now_bleu = brev_penalty * math.exp(logsum / ngram)
	return now_bleu

if __name__ == '__main__':
    s1 = '1234123412341233412341234123412340'
    w1 = [int(i) for i in list(s1)]
    s2 = '2341231234123412312312313123443130'
    w2 = [int(i) for i in list(s2)]
    ref, lens = getRefDict(w1[:-1], 4)
    print calBleu(w2[:-1],ref,lens,4)[0]
