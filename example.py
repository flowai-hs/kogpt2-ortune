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
while 1:
  input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
  pred = model(input_ids)[0]
  last_pred = last_pred = pred.squeeze()[-1]
  gen = sampling.top_k(last_pred, vocab, 5)
  if gen == '</s>':
      break
  sent += gen.replace('▁', ' ')
  toked = tok(sent)
print(sent)
