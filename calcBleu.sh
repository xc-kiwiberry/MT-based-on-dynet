#!/bin/bash
perl ~/projects/dynet/examples/cpp/encdec/multi-bleu.perl \
     ~/data_set/dev_test/nist06/nist06.en \
     < test_big_13_nist06_1_620_1000.out
     #> result_5.txt

