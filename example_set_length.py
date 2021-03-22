import torch
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
import sampling

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path)


sent = input('sent: ')
toked = tok(sent)
# 토큰 생성 카운트
count = 0
# 기본은 </s>에서 break, 원하는 길이가 있을 경우 output사이즈 수정
output_size = 32

while 1:
  input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
  pred = model(input_ids)[0]
  last_pred = pred.squeeze()[-1]
  gen = sampling.top_k(last_pred, vocab, 5)
  
  # 원하는 길이 문자열을 생성하고 싶으면 이 부분 주석 해제
  if gen == '</s>':
    sent += gen.replace('▁', ' ')
    toked = tok(sent)
    count = 0
    break
  
  sent += gen.replace('▁', ' ')
  toked = tok(sent)
  count += 1

print(sent)