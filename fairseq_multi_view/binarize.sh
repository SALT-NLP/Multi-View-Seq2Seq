# Change the name of source file for pre-processing, as well as the destdir

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "../data/train_sent_c99_label.bpe" \
  --validpref "../data/val_sent_c99_label.bpe" \
  --destdir "cnn_dm-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
  
  
  
# Change the name of source file for pre-processing, as well as the destdir

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "../data/train_sent_trans_cons_label.bpe" \
  --validpref "../data/val_sent_trans_cons_label.bpe" \
  --destdir "cnn_dm-bin_2/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;