name: inverse
class: center, middle, inverse
layout: true
title: Python-basic

---
class: titlepage, no-number

# Machine Learning with MATLAB
## .gray.author[Youngjae Yu]

### .x-small[https://github.com/yj-yu/matlab-ml]
### .x-small[https://yj-yu.github.io/matlab-ml]

.bottom.img-66[ ![](images/snu-logo.png) ]

---
layout: false

## About

- LSTM을 이용한 시퀀스 데이터 분류
- 스마트폰으로부터 얻은 센서 데이터로 동작 인식하기
- 일기예보 텍스트 데이터 분류하기

---

template: inverse

# MATLAB Deep Learning Toolbox

Long Short Term Memory Network


---

## LSTM

장단기 기억(LSTM) 네트워크를 사용하여 분류 및 회귀 작업에 대해 시퀀스 및 시계열 데이터로 작업 가능

- LSTM 신경망은 시퀀스 데이터의 시간 스텝 간의 장기적인 종속성을 학습할 수 있는 순환 신경망(RNN)의 일종

LSTM 네트워크 구조

- Sequence Input 계층은 네트워크에 시퀀스 또는 시계열 데이터를 입력합니다. 
- LSTM 계층은 시퀀스 데이터의 시간 스텝 간의 장기 종속성을 학습합니다.

.center.img-66[ ![](images/img1.png) ]

- 연속적인 회귀 예측도 가능합니다.

.center.img-50[ ![](images/img2.png) ]

---

## LSTM 아키텍처

다음 도식은 길이가 S인 특징(채널) C개를 갖는 시계열 X가 LSTM 계층을 통과하는 흐름을 보여줍니다.
- 이 도식에서 $h_t$ 와 $c_t$ 는 각각 시간 스텝 t에서의 출력값(은닉(hidden) 상태라고도 함)과 셀 상태를 나타냅니다.

.center.img-66[ ![](images/img0.png) ]

---

## LSTM 아키텍처

첫 번째 LSTM 블록은 네트워크의 초기 상태와 시퀀스의 첫 번째 시간 스텝을 사용하여 첫 번째 출력값과 업데이트된 셀 상태를 계산합니다. 
- (repeat) 시간 스텝 $t$에서, 이 블록은 네트워크의 현재 상태 $(t-1)$ 와 시퀀스의 다음 스텝 $(x)$ 을 사용하여 출력값과 업데이트된 셀 상태 $c_t$를 계산합니다.


.center.img-66[ ![](images/img0.png) ]


---

## LSTM 아키텍처

입력 게이트(i) : 셀 상태 업데이트의 수준 제어

망각 게이트(f) : 셀 상태 재설정(망각)의 수준 제어

셀 후보(g) : 셀 상태에 정보 추가

출력 게이트(o) : 은닉 상태에 추가되는 셀 상태의 수준 제어

.center.img-66[ ![](images/cell.png) ]

```python
layer = lstmLayer(numHiddenUnits)
layer = lstmLayer(numHiddenUnits,Name,Value)
```


---

## LSTM 네트워크 (분류)

시퀀스 입력 계층의 크기를 입력 데이터의 특징 개수로 설정합니다. 
- FC layer 의 크기를 클래스 개수로 설정합니다. 
- 시퀀스 길이는 따로 지정할 필요가 없습니다.
- LSTM 계층의 경우, 은닉 유닛의 개수와 출력 모드 'last'를 지정합니다.

```python
numFeatures = 12;
numHiddenUnits = 100;
numClasses = 9;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
```

---

## LSTM 네트워크 (분류)

sequence-to-sequence 분류를 위한 LSTM 네트워크를 만들려면 sequence-to-label 분류와 동일한 아키텍처를 사용하되 LSTM 계층의 출력 모드를 'sequence'로 설정합니다.

```python
numFeatures = 12;
numHiddenUnits = 100;
numClasses = 9;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
```

---

## LSTM 네트워크 (회귀)

sequence-to-one 회귀를 위한 LSTM 네트워크를 만들려면 시퀀스 입력 계층, LSTM 계층, 완전 연결 계층, 회귀 출력 계층을 포함하는 계층 배열을 만듭니다.

LSTM 계층의 경우, 은닉 유닛의 개수와 출력 모드 'last'를 지정합니다.

```python
numFeatures = 12;
numHiddenUnits = 125;
numResponses = 1;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numResponses)
    regressionLayer];
```

---

## LSTM 네트워크 (회귀)

sequence-to-sequence 회귀를 위한 LSTM 네트워크를 만들려면 sequence-to-one 회귀와 동일한 아키텍처를 사용하되 LSTM 계층의 출력 모드를 'sequence'로 설정합니다.

```python
numFeatures = 12;
numHiddenUnits = 125;
numResponses = 1;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];
```


---

## Deep LSTM

LSTM 계층 앞에 출력 모드가 'sequence'인 LSTM 계층을 추가로 삽입하여 LSTM 네트워크의 심도를 높일 수 있습니다. 과적합을 방지하기 위해 LSTM 계층 뒤에 드롭아웃 계층을 삽입할 수 있습니다.
- lstmLayer(numHiddenUnits1,'OutputMode','sequence') 추가

sequence-to-label 분류 네트워크의 경우, 마지막 LSTM 계층의 출력 모드가 'last'가 되어야 합니다.

```python
numFeatures = 12;
numHiddenUnits1 = 125;
numHiddenUnits2 = 100;
numClasses = 9;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits2,'OutputMode','last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
```

---

## LSTM 으로 시퀀스 데이터  분류

Japanese Vowels 데이터 세트를 사용합니다.

연속해서 발화된 2개의 일본어 모음을 나타내는 시계열 데이터를 주고 화자를 인식하도록 LSTM 네트워크를 훈련시킵니다. 
- 훈련 데이터는 화자 9명의 시계열 데이터를 포함합니다. 
- 각 시퀀스는 12개의 특징을 가지며 길이가 서로 다릅니다. 
- 데이터 세트는 270개의 훈련 관측값과 370개의 테스트 관측값을 포함합니다.

```python
[XTrain,YTrain] = japaneseVowelsTrainData;
XTrain(1:5)
```
- XTrain의 요소는 각 특징에 대해 하나의 행을 갖는 12개의 행과 각 시간 스텝에 대해 하나의 열을 갖는 가변 개수의 열로 이루어진 행렬입니다.
- Y는 9명의 화자 각각에 대응하는 레이블 "1","2",...,"9"로 구성된 categorical형 벡터입니다. 



---

## LSTM 으로 시퀀스 데이터  분류

첫 번째 시계열을 플롯으로 시각화합니다. 선은 각각 하나의 feature에 대응됩니다.

```python
figure
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation 1")
numFeatures = size(XTrain{1},1);
legend("Feature " + string(1:numFeatures),'Location','northeastoutside')
```


---

## LSTM 으로 시퀀스 데이터  분류

기본적으로 훈련 중에 훈련 데이터가 미니 배치로 분할되고 모든 시퀀스의 길이가 같아지도록 시퀀스가 채워집니다. 
- 0으로 채우는 것이 많을수록 학습에는 좋을게 없습니다.

시퀀스 데이터를 정렬한 다음 미니 배치 크기를 선택하여 하나의 미니 배치에 속한 시퀀스들이 비슷한 길이를 갖도록 합니다. 
- 그림에서 데이터를 정렬하기 전과 후의 시퀀스 채우기의 효과를 보여줍니다.

.center.img-66[ ![](images/img3.png) ]

---

## LSTM 으로 시퀀스 데이터  분류

각 관측값에 대한 시퀀스 길이를 가져옵니다.
```python
numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end
```

시퀀스 길이를 기준으로 데이터를 정렬합니다.
```python
[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);
```

---

## LSTM 으로 시퀀스 데이터  분류

정렬된 시퀀스 길이를 막대 차트로 표시합니다.

```python
figure
bar(sequenceLengths)
ylim([0 30])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")
```
.center.img-50[ ![](images/img4.png) ]


---

## LSTM 으로 시퀀스 데이터  분류

미니 배치 크기를 27로 선택하여 훈련 데이터를 균등하게 나누고 미니 배치에 채워지는 양을 줄입니다. 

LSTM 네트워크 아키텍처를 정의합니다. 시퀀스의 입력 크기를 12(입력 데이터의 차원)로 지정합니다. 
- 은닉 유닛 100개를 갖는 LSTM 계층을 지정하고 시퀀스의 마지막 요소를 출력하게 합니다. 

```python
miniBatchSize = 27;

inputSize = 12;
numHiddenUnits = 100;
numClasses = 9;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
```

---

## LSTM 으로 시퀀스 데이터  분류

이번에는 훈련 옵션을 지정합니다. 
- 솔버를 'adam'으로 지정하고, 기울기 임계값을 1로 지정하고, 최대 Epoch 횟수를 100으로 지정합니다. 

```python
maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
```

---

## LSTM 으로 시퀀스 데이터 분류

trainNetwork를 사용하여 지정된 훈련 옵션으로 LSTM 네트워크를 훈련시킵니다.

```python
net = trainNetwork(XTrain,YTrain,layers,options);
```

.center.img-66[ ![](images/img5.png) ]


---

## LSTM 으로 시퀀스 데이터 분류

Japanese Vowels 테스트 데이터를 불러옵니다. 
- XTest는 12개 차원으로 된 서로 다른 길이의 시퀀스 370개를 포함하는 셀형 배열입니다. 
- YTest는 9명의 화자에 대응하는 레이블 "1","2",..."9"로 구성된 categorical형 벡터입니다.

```python
[XTest,YTest] = japaneseVowelsTestData;
XTest(1:3)
```

---

## LSTM 으로 시퀀스 데이터 분류

LSTM 네트워크 net은 비슷한 길이를 갖는 시퀀스를 포함하는 미니 배치를 사용하여 훈련되었습니다.
- 테스트 데이터도 같은 방식으로 구성되도록 합니다. 
- 시퀀스 길이를 기준으로 테스트 데이터를 정렬합니다.

```python
numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);
```

---

## LSTM 으로 시퀀스 데이터 분류

테스트 데이터를 분류합니다. 
- 훈련 데이터와 동일한 양의 채우기를 적용하려면 시퀀스 길이를 'longest'로 지정합니다.

```python
miniBatchSize = 27;
YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

acc = sum(YPred == YTest)./numel(YTest)
```

---

## 시계열 예측

미래 추세를 예측

.center.img-33[ ![](images/img8.png) ]

LSTM으로 분류 뿐만 아니라 시계열 예측이 가능합니다.

.center.img-33[ ![](images/img2.png) ]

---

## LSTM 으로 시계열 예측

이번 실습에서는 chickenpox_dataset를 사용합니다. 
- 이전 달들의 수두 발병 건수를 주고 수두 발병의 건수를 예측하도록 LSTM 네트워크를 훈련시킵니다.

chickenpox_dataset는 시간 스텝이 달에 대응되고 값이 발병 건수에 대응되는 하나의 시계열을 포함합니다. 
- 출력값은 각 요소가 하나의 시간 스텝인 셀형 배열입니다. 
- 데이터가 행 벡터가 되도록 형태를 변경합니다.

```python
data = chickenpox_dataset;
data = [data{:}];

figure
plot(data)
xlabel("Month")
ylabel("Cases")
title("Monthy Cases of Chickenpox")
```



---

## LSTM 으로 시계열 예측

훈련 데이터와 테스트 데이터를 나눕니다. 시퀀스의 처음 90%에 대해 훈련을 진행하고 마지막 10%에 대해 테스트를 진행합니다.
```python

numTimeStepsTrain = floor(0.9*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);
```

훈련의 발산을 방지하기 위해 평균 0, 분산 1이 되도록 훈련 데이터를 표준화합니다. 
- 테스트 데이터는 훈련 데이터와 동일한 파라미터를 사용하여 예측 시점에 표준화해야 합니다.

```python
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;
```

---

## LSTM 으로 시계열 예측

예측 변수와 응답 변수 준비하기
시퀀스의 미래의 시간 스텝 값을 예측하려면 응답 변수가 값이 시간 스텝 하나만큼 이동된 훈련 시퀀스가 되도록 지정
- 즉, LSTM 네트워크는 입력 시퀀스의 각 시간 스텝마다 다음 시간 스텝의 값을 예측하도록 학습합니다.

```python
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
```

LSTM 회귀 네트워크를 만듭니다. LSTM 계층이 200개의 은닉 유닛을 갖도록 지정합니다.

```python
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
```

---

## LSTM 으로 시계열 예측

훈련 옵션을 지정합니다. 솔버를 'adam'으로 지정하고 250회의 Epoch에 대해 훈련시킵니다. 

```python
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
```
trainNetwork를 사용하여 지정된 훈련 옵션으로 LSTM 네트워크를 훈련시킵니다.

```python
net = trainNetwork(XTrain,YTrain,layers,options);
```

---

## LSTM 으로 시계열 예측

.center.img-80[ ![](images/img7.png) ]

---

## LSTM 으로 시계열 예측

미래의 여러 시간 스텝의 값을 예측하려면 predictAndUpdateState 함수를 사용하여 한 번에 하나의 시간 스텝을 예측한 다음 각 예측에 대해 네트워크 상태를 업데이트
- 각 예측에서 직전의 예측을 함수의 입력값으로 사용합니다.

훈련 데이터와 동일한 파라미터를 사용하여 테스트 데이터를 표준화합니다.

```python
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
```

---

## LSTM 으로 시계열 예측

네트워크 상태를 초기화하려고 먼저 훈련 데이터 XTrain에 대해 예측을 수행했습니다. 

다음으로, 훈련 응답 변수 YTrain(end)의 마지막 시간 스텝을 사용하여 첫 번째 예측을 수행합니다. 

루프를 사용해 나머지 예측을 반복 수행하고 직전 예측을 predictAndUpdateState의 입력값으로 사용합니다.

```python
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
```

---

## LSTM 으로 시계열 예측

앞에서 계산한 파라미터를 사용하여 예측의 표준화를 해제합니다.
```python
YPred = sig*YPred + mu;
```

훈련 진행 상황 플롯은 표준화된 데이터로부터 계산된 RMSE(제곱평균제곱근 오차)를 보고합니다. 표준화를 해제한 예측으로부터 RMSE를 계산합니다.
```python
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))
```

---

## LSTM 으로 시계열 예측

예측된 값을 사용하여 훈련 시계열을 플로팅합니다.
```python
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])
```

.center.img-33[ ![](images/img8.png) ]

---

## LSTM 으로 시계열 예측

예측된 값과 테스트 데이터를 비교합니다.

```python
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
```

---

## LSTM 으로 시계열 예측

.center.img-80[ ![](images/img9.png) ]


---

template: inverse

# 일기예보 텍스트 데이터 분류하기

Long Short Term Memory Network for NLP


---

## 학습목표

- 데이터를 가져오고 전처리합니다.

- 단어 인코딩을 사용하여 단어를 숫자 시퀀스로 변환합니다.

- 단어 임베딩 계층을 사용하여 LSTM 네트워크를 만들고 훈련시킵니다.

- 훈련된 LSTM 네트워크를 사용하여 새로운 텍스트 데이터를 분류합니다.

---

## Text 분류하기

심층 학습 장단기 기억(LSTM) 네트워크를 사용하여 일기 예보의 텍스트 설명을 분류하는 방법입니다.

- LSTM 네트워크에 텍스트를 입력하려면 먼저 텍스트 데이터를 숫자형 시퀀스로 변환
- 숫자형 인덱스 시퀀스로 매핑하는 단어 인코딩을 사용하면 됩니다.
- 더 나은 결과를 위해 단어 임베딩 사용. 단어 임베딩은 어휘에 있는 단어를 스칼라형 인덱스가 아닌 숫자형 벡터로 매핑합니다. 
- 임베딩은 비슷한 의미를 갖는 단어들이 비슷한 벡터를 갖도록 단어의 의미 체계 정보를 캡처합니다. 벡터 연산을 통해 단어 사이의 관계도 모델링합니다. 
- 예를 들어, "왕 대 여왕은 남자 대 여자와 같다"라는 관계는 왕 – 남자 + 여자 = 여왕이라는 식으로 설명됩니다.


---

## 데이터 가져오기

일기 예보 데이터를 가져옵니다. 
- 날씨 이벤트에 대한 텍스트로 된 설명을 포함합니다. 
- 텍스트 데이터를 문자열로 가져오도록 텍스트 유형을 'string'으로 지정하십시오.

```python
filename = "weatherReports.csv";
data = readtable(filename,'TextType','string');
head(data)
```

---

## Text 분류하기

일기 예보가 비어 있는 행은 테이블에서 제거합니다.
```python
idxEmpty = strlength(data.event_narrative) == 0;
data(idxEmpty,:) = [];
```
이 예제의 목표는 event_type 열의 레이블을 기준으로 이벤트를 분류하는 것입니다. 데이터를 클래스별로 나누기 위해 레이블을 categorical형으로 변환합니다.
```python
data.event_type = categorical(data.event_type);
```

---

## Text 분류하기

히스토그램을 사용하여 데이터의 클래스 분포를 표시합니다. 

```python
f = figure;
f.Position(3) = 1.5*f.Position(3);

h = histogram(data.event_type);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")
```

.center.img-66[ ![](images/img10.png) ]


---

## Text 분류하기

많은 클래스가 적은 수의 관측값을 포함하고 있어서 데이터의 클래스 간에 균형이 맞지 않습니다. 
- 이런 식으로 클래스 간에 균형이 맞지 않으면 네트워크가 덜 정확한 모델로 수렴할 수 있습니다. 
- 한가지 대응책으로, 10회보다 적게 나타나는 클래스를 모두 제거합니다.

히스토그램에서 클래스의 빈도 수와 클래스 이름을 가져옵니다.

```python
classCounts = h.BinCounts;
classNames = h.Categories;
```

관측값이 10개보다 작은 클래스를 찾습니다.

```python
idxLowCounts = classCounts < 10;
infrequentClasses = classNames(idxLowCounts)
```

---

## Text 분류하기

빈도가 적은 이러한 클래스를 데이터에서 제거합니다. removecats를 사용하여 categorical형 데이터에서 사용되지 않는 범주를 제거합니다.

```python
idxInfrequent = ismember(data.event_type,infrequentClasses);
data(idxInfrequent,:) = [];
data.event_type = removecats(data.event_type);
```

데이터는 이제 적당한 크기의 클래스로 정렬되어 있습니다. 

다음 단계는 데이터를 훈련 세트, 검증 세트, 테스트 세트로 분할합니다.
- 데이터를 훈련 파티션, 그리고 검증과 테스트를 위한 홀드아웃 파티션으로 분할합니다. 홀드아웃 백분율을 30%로 지정합니다.
```python
cvp = cvpartition(data.event_type,'Holdout',0.3);
dataTrain = data(training(cvp),:);
dataHeldOut = data(test(cvp),:);
```

---

## Text 분류하기

홀드아웃 세트를 다시 분할하여 검증 세트를 얻습니다. 홀드아웃 백분율을 50%로 지정합니다. 이렇게 하면 훈련 관측값 70%, 검증 관측값 15%, 테스트 관측값 15%로 데이터가 분할됩니다.
```python
cvp = cvpartition(dataHeldOut.event_type,'HoldOut',0.5);
dataValidation = dataHeldOut(training(cvp),:);
dataTest = dataHeldOut(test(cvp),:);
```
분할된 테이블에서 텍스트 데이터와 레이블을 추출합니다.
```python
textDataTrain = dataTrain.event_narrative;
textDataValidation = dataValidation.event_narrative;
textDataTest = dataTest.event_narrative;
YTrain = dataTrain.event_type;
YValidation = dataValidation.event_type;
YTest = dataTest.event_type;
```

---

## Text 분류하기

데이터를 올바르게 가져왔는지 확인을 위해 훈련 텍스트 데이터를 시각화합니다.

```python
figure
wordcloud(textDataTrain);
title("Training Data")
```
.center.img-66[ ![](images/img11.png) ]


---

## 텍스트 데이터 전처리하기

텍스트 데이터를 토큰화하고 전처리하는 함수를 만듭니다. 이 예제의 마지막에 나오는 함수 preprocessText는 다음 단계를 수행합니다.

- tokenizedDocument를 사용하여 텍스트를 토큰화합니다.

- lower를 사용하여 텍스트를 소문자로 변환합니다.

- erasePunctuation을 사용하여 문장 부호를 지웁니다.

preprocessText 함수를 사용하여 훈련 데이터와 검증 데이터를 전처리합니다.
```python
documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation)
```

확인해봅시다.
```python
documentsTrain(1:5)
```

---

## Text 분류하기

문서를 LSTM 네트워크에 입력하려면 단어 인코딩을 사용하여 문서를 숫자형 인덱스로 구성된 시퀀스로 변환하십시오.

단어 인코딩을 만들려면 wordEncoding 함수를 사용하십시오.

```python
enc = wordEncoding(documentsTrain);
```

다음 변환 단계는 문서가 모두 같은 길이가 되도록 채우고 자릅니다.

먼저 목표 길이를 선택하고, 목표 길이보다 긴 문서는 자르고 목표 길이보다 짧은 문서는 왼쪽을 채웁니다. 
최상의 결과를 위해 목표 길이는 다량의 데이터가 버려지지 않을 만큼 짧아야 합니다. 

```python
documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")
```

---

## Text 분류하기

histogram을 보시면, 대부분의 훈련 문서가 75개 미만의 토큰을 갖습니다. 
- 이 값을 자르기와 채우기의 목표 길이로 사용합니다.

doc2sequence를 사용하여 문서를 숫자형 인덱스로 구성된 시퀀스로 변환합니다. 
- 시퀀스의 길이가 75가 되도록 자르거나 왼쪽을 채우려면 'Length' 옵션을 75로 설정

```python
XTrain = doc2sequence(enc,documentsTrain,'Length',75);
XTrain(1:5)
```


---

## Text 분류하기

Sequence-to-label 분류 문제에서 LSTM 계층을 사용하려면 출력 모드를 'last'로 설정
- 마지막으로, 클래스 개수와 동일한 크기를 갖는 완전 연결 계층, 소프트맥스 계층, 분류 계층을 추가합니다.

```python
inputSize = 1;
embeddingDimension = 100;
numWords = enc.NumWords;
numHiddenUnits = 180;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
```

---

## Text 분류하기

- 훈련 진행 상황을 모니터링하려면 'Plots' 옵션을 'training-progress'로 설정
- 'ValidationData' 옵션을 사용하여 검증 데이터를 지정
- 세부 정보가 출력되지 않도록 하려면 'Verbose'를 false로 설정

```python
options = trainingOptions('adam', ...
    'MaxEpochs',10, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);
```

trainNetwork 함수를 사용하여 LSTM 네트워크를 훈련시킵니다.
```python
net = trainNetwork(XTrain,YTrain,layers,options);
```

---

## LSTM 네트워크 테스트하기

전처리된 테스트 데이터에 대해 훈련된 LSTM 네트워크 net을 사용하여 예측을 수행합니다.
- 테스트 데이터를 전처리합니다.
- doc2sequence를 사용하여 테스트 문서를 시퀀스로 변환합니다.

```python
textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);
documentsTest = erasePunctuation(documentsTest);
XTest = doc2sequence(enc,documentsTest,'Length',75);
XTest(1:5)
```

---

## LSTM 네트워크 테스트하기

훈련된 LSTM 네트워크를 사용하여 테스트 문서를 분류합니다.

```python
YPred = classify(net,XTest);
accuracy = sum(YPred == YTest)/numel(YPred)
```

---

## LSTM 네트워크 테스트하기

새 일기 예보를 포함하는 string형 배열을 만듭니다.

```python
reportsNew = [ ...
    "Lots of water damage to computer equipment inside the office."
    "A large tree is downed and blocking traffic outside Apple Hill."
    "Damage to many car windshields in parking lot."];
documentsNew = preprocessText(reportsNew);
XNew = doc2sequence(enc,documentsNew,'Length',75);
```

훈련된 LSTM 네트워크를 사용하여 새 시퀀스를 분류합니다.

```python
[labelsNew,score] = classify(net,XNew);
```

---

## LSTM 네트워크 테스트하기

결과를 눈으로 확인합니다.

```python
[reportsNew string(labelsNew)]
```



---
name: last-page
class: center, middle, no-number
## Thank You!


<div style="position:absolute; left:0; bottom:20px; padding: 25px;">
  <p class="left" style="margin:0; font-size: 13pt;">
</div>

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]




<!-- vim: set ft=markdown: -->
