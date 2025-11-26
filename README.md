# PyTorch Machine Learning Practice

## 1. Project Overview
**PyTorch를 활용하여 데이터 전처리부터 모델 학습, 성능 평가까지 머신러닝의 전체 파이프라인(End-to-End)을 직접 구현한 프로젝트입니다.**

단순히 만들어진 모델을 사용하는 것을 넘어, **데이터가 어떻게 가공되어 학습되는지 그 내부 동작 원리를 이해**하고자 진행했습니다. 결측치가 포함된 Raw Data를 정제하고, `nn.Module`을 활용해 모델을 직접 설계하여 예측 시스템을 구축했습니다.

* **Goal:** PyTorch 기반의 커스텀 모델 설계 및 데이터 처리 역량 확보
* **Tech Stack:** Python, PyTorch, Pandas, Scikit-learn
* **Tasks:**
    * 타이타닉 생존자 예측 (Classification)
    * 캘리포니아 집값 예측 (Regression)
    * 당뇨병 진단 예측 (Classification)

## 2. Key Features & Implementation
데이터를 단순히 모델에 넣는 것이 아니라, **학습 성능을 극대화하기 위한 전처리 및 최적화 과정**에 집중했습니다.

### 📊 데이터 전처리 (Data Preprocessing)
* **결측치 처리 (Handling Missing Values):** 데이터 손실을 막기 위해 단순히 행을 삭제하지 않고, 통계적 최빈값 등으로 빈 곳을 채워 학습 데이터를 확보했습니다.
* **데이터 변환 (Encoding):** 컴퓨터가 이해할 수 있도록 성별(0/1)과 같은 범주형 데이터를 **One-hot Encoding** 등으로 변환하여 학습 효율을 높였습니다.
* **정규화 (Normalization):** 데이터 간의 스케일 차이로 인한 학습 저하를 막기 위해 `StandardScaler`를 적용하여 데이터 범위를 통일했습니다.

### 🚀 모델링 및 최적화 (Modeling & Optimization)
* **모델 모듈화:** `nn.Module`을 상속받은 커스텀 클래스를 설계하여 코드의 재사용성을 높이고 관리를 용이하게 했습니다.
* **효율적인 학습:** 대용량 데이터 처리를 고려하여 `DataLoader`를 도입하고, 데이터를 소분하여 학습하는 **Mini-batch(Batch size: 96)** 방식을 구현했습니다.

## 3. Performance Results
구현한 모델의 성능을 정량적으로 평가하고 분석했습니다.

* **Titanic Survival Prediction:**
    * **정확도(Accuracy):** **81.56%** 달성 
    * **분석:** Confusion Matrix를 통해 모델의 오답 패턴(FP, FN)을 상세하게 점검했습니다.
* **California Housing Price Prediction:**
    * **오차율(MSE):** **0.3819** 기록 
    * **성과:** 정규화(Normalization) 적용 전후의 Loss 감소 폭을 비교하여 전처리의 중요성을 검증했습니다.

## 4. Project Structure
```text
├── data/                      # Dataset files (csv, pkl)
├── Own_ML_Class.py            # Custom Model Classes (Inherits nn.Module)
├── titanic_logistic_reg...py  # Classification Task (Titanic)
├── diabetes_logistic_reg...py # Classification Task (Diabetes)
└── calhouse_linear_reg...py   # Regression Task (California Housing)
