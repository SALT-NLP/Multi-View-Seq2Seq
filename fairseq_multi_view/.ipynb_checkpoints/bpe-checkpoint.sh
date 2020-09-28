# Change the name of source file for pre-processing

for SPLIT in train_sent_c99_label val_sent_c99_label test_sent_c99_label
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "../data/$SPLIT.$LANG" \
    --outputs "../data/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done



for SPLIT in train_sent_trans_cons_label val_sent_trans_cons_label test_sent_trans_cons_label
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "../data/$SPLIT.$LANG" \
    --outputs "../data/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done