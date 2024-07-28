# Unveiling Context Control Mechanism in Large Language Models by Layer-wise Evaluation
[석사 학위 논문 ](https://github.com/Songwooseok123/Master-s-thesis/blob/main/%EC%86%A1%EC%9A%B0%EC%84%9D_%EC%84%9D%EC%82%AC%ED%95%99%EC%9C%84%EC%B2%AD%EA%B5%AC%EB%85%BC%EB%AC%B8%20(1).pdf)관련 연구입니다.

## 결론 
![image](https://github.com/Songwooseok123/Master-s-thesis/assets/80091008/50b5da15-426d-4054-b19e-5e468e779d07)

- Dialogue Context 앞에 감정을 제어하는 prompt를 추가해서, zero-shot으로 감정 controll generation을 하는 상황
- 결과1: Layer 관점에서 현상을 관찰한 결과, 감정을 제어하는 Prompt를 입력으로 주어 Zero-shot learning을 수행하는 메커니즘이 언어 모델 전체를 Fine-tuning한 것과 비슷한 양상을 보인다.  
- 결과2: Dialogue Context에 감정 정보를 빠르게(더 앞선 레이어에서) 인코딩 할 수록, 레이어 관점에서 감정을 빨리(더 앞선 레이어에서) 구분할 수록, 감정 제어를 잘한다!

## Introduction
- LLM Zero shot 잘 함
- Controlled Dialogue Generation(CDG)
  - Dialogue Context가 모델에 입력되었을 때, 다음 발화를 생성하는 task 수행 중, 사용자가 정의한 속성을 가진 발화를 생성하게 하는 task
  - 모델 별로 속성 제어 성능이 차이가 남
  - 레이어 관점에서 왜 이런 차이가 생기는지, 속성을 어디에 Encoding하는지 보는 연구

## Emotion Controlled Dialogue Generation
- 감정을 제어하는 Dialogue Generation
  - 감정을 제어하기 위해 모델별로 정해진 Prompt Template을 사용함(Emotion description)
    - Prompt Template은 모델이 학습된 방법에 따라 다르게 정의됨
  - 감정 제어을 했을 때(b)와 안 했을 때(a)의 결과를 비교
  - ![image](https://github.com/Songwooseok123/Master-s-thesis/assets/80091008/10cb82da-16cd-487e-af9e-5f4297433442)

  - ![image](https://github.com/Songwooseok123/Master-s-thesis/assets/80091008/c0434aca-305c-495e-a43f-acc01e6bb6c1)

## Effects of Emotion Control Instruction on Dialogue Contexts
- 위의 결과를 해석하는 Section
  - 밑의 실험 3개를 통해서 결론을 도출함.
### 1. Cosine Similarity of Hidden states of Dialogue Context with and without Emotion Control
  - ![image](https://github.com/Songwooseok123/Master-s-thesis/assets/80091008/c1d555db-5ede-4f92-ab17-a4a46358c123)
### 2. 2D Visualization through PCA
  - ![image](https://github.com/Songwooseok123/Master-s-thesis/assets/80091008/d8e53a58-e748-4177-94ec-0a0ceeea1689)
### 3. Probing with Logistic Regression Classifier
  - ![image](https://github.com/Songwooseok123/Master-s-thesis/assets/80091008/26904a66-27c2-4011-acc8-cc50adc40442)
### 최종 결과 표 
- ![image](https://github.com/Songwooseok123/Master-s-thesis/assets/80091008/d689aff8-7858-4df0-9245-85d8243ede34)




## 코드 설명
- **get_model.py**
  - huggingface pre_trained 모델 불러 오는 코드(모델 양자화 포함)
- **get_generated_sentences.ipynb**
  - 모델 별로 주어진 dialogue context를 가공하여 prompt template을 만들고 발화 생성
  - 생성된 발화의 감정 정확도를 Bert Classifier로 평가
- **get_hidden_states_code.ipynb**
  - 모델 별로 dialogue context의 (layer별)hidden states를 뽑아내는 코드 
- **Paper_results.ipynb**, **Paper_figures.ipynb**
  - 감정 제어를 했을 때와 안 했을 때의 hidden states의 cosine 유사도 그래프 
  - hidden state를 PCA로 차원 축소 후 시각화 
  - hidden state layer별 probing
    - Logistic Regression 정확도 그래프
  - Cosine 유사도의 Plateau 지점과 속성 정확도의 관계 그래프
  - Logistic Regression 수렴 지점과 속성 정확도의 관계 그래프

