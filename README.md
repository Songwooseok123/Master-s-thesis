# Master-s-thesis
석사 학위 논문 관련 연구입니다.

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

