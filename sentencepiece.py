import sentencepiece as spm

# train sentencepiece model from 'ml-1m-seq.txt` and makes `ml-1m-seq.model` and `ml-1m-seq.vocab`

prefix = 'ml-1m-seq'
corpus = './data/' + prefix + '.txt'

vocab_size = 3420
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" + # 문장 최대 길이
    " --user_defined_symbols=<cls>,<sep>") # 사용자 정의 토큰

# makes segmenter instance and loads the model file (ml-1m-seq.model)
sp = spm.SentencePieceProcessor()
sp.load(prefix + '.model')

# encode: text => id
print(sp.encode_as_pieces('1 123 543 123 43 3789'))
print(sp.encode_as_ids('1 123 543 123 43 3789'))

# decode: id => text
print(sp.decode_pieces(['▁1', '▁123', '▁543', '▁123', '▁43', '▁37', '89']))
print(sp.decode_ids(sp.encode_as_ids('1 123 543 123 43 3789')))