
## KoGPT2 (한국어 GPT-2)

### Why'?'

* OpenAI GPT-2 모델의 한국어 성능 한계

### Model

* `GPT-2 base` 모델

```python
GPT2Model(units=768,
    max_length=1024,
    num_heads=12,
    num_layers=12,
    dropout=0.1,
    vocab_size=50000)
```

* `Fused GELU`를 기반으로 10% 이상의 학습 및 추론 속도 향상

#### Tokenizer

* 2천 5백만 이상의 문장으로 학습(wiki + news)
* BPE(Byte Pair Encoding)
* 50,000 토큰

#### Data

| Data  |  # of Sentences  | # of Words |
|---|---|---|
| Korean Wiki  |  5M |  54M  |
| Korean News  |  120M | 1.6B |
| Other corpus |   9.4M, 18M | 88M, 82M |

* 원시 문장 (Raw text) 기준 약 20GB의 데이터 사용


#### How to install

```sh
git clone https://github.com/flowai-hs/kogpt2-org
cd kogpt2-org
pip install -r requirements.txt
pip install .
```


##### Requirements

* torch === 1.5.1
* mxnet == 1.6.0
* gluonnlp >= 0.8.3
* sentencepiece >= 0.1.6
* transformers >= 2.1.1


### How to use

```sh
python example.py
```
또는 
```sh
python example_set_length.py
```
문장을 입력한다.
ex)  sent: 오늘 점심


### Contacts

`KoGPT2` 관련 이슈는 [이곳](https://github.com/SKT-AI/KoGPT2/issues)에 올려주세요.

### License

`KoGPT2`는 `modified MIT` 라이선스 하에 공개되어 있습니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 `LICENSE` 파일에서 확인하실 수 있습니다.
