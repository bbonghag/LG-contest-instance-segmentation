## LG화학 입자 형태 분석 모델 개발 해커톤 - DeepDream Rank7🏅🎖

### 입자 검출 정보를 기반으로 입자들의 형태 변화를 계량적 지표로 산출 가능한 모델 개발 <br/><br/>

<img src="https://user-images.githubusercontent.com/103362361/187381517-1abe006c-6073-4f27-9e55-ae35d388828e.png"  width="400" height="200"/>

---

### Team 
- DeepDream 

### Challenge Link  
- https://aifactory.space/competition/detail/2067 

### Project period   
- 2022.07.07 - 2022.08.08

### Host & Organizer 
- LG화학 & AIFactory

### Description
- 유체상에 떠다니는 입자를 촬영한 화상을 바탕으로 각 입자와 그 형상을 최대한 잘 검출해내는 Instance Segmentation 모델 개발.

--- 
### Data
- LG화학에서 제공하는 유체상에 떠다니는 입자를 촬영한 사진.
- train dataset 520장, Test dataset 350장 및 coco dataset 형식의 어노테이션 파일(입자 레이블링 형식에 따라 label_train.json, label(polygon)train.json) 
- 객체 카테고리는 1개(Normal) 클래스만 존재, 이미지 해상도는 (Height, Width) = (1024, 1280) 크기


### Progress
- 1주차(7/13 ~ 7/17)  
  - Instance segmentation 공부, MMdetection 라이브러리 사용법 익히기, base-line 돌려보기, EDA(데이터분석) 
- 2주차(7/18 ~ 7/24)  
  - Segmentation model 조원들에게 분배후 제출하여 점수가 높은 모델들 선정 후 model 공부, modeling
- 3주차(7/25 ~ 7/31)  
  - 전처리 
    - Transform(이미지중 일부를 변형하여 학습에 사용)  
    - Augmentation(이미지 증강) 리스트업하여 각각 어떤 기법을 사용할 것인지 분배하고 성능확인 및 어떤걸 쓸지 선정.
    - backbone 분배후 성능확인 및 backbone 선정 
- 4주차(8/1 ~ 8/8)  
  - Optimizer 분배 후 선정, 성능향상을 위한 하이퍼파라미터 조정.


### Models  
- SCNet, Mask R-CNN, Mask Scoring R-CNN, Cascade Mask R-CNN
  

### Result
- 62팀 중 7등 달성!
 
  <img src="https://user-images.githubusercontent.com/103362361/187385788-913ff59d-cc4a-4d4a-bebc-456c99575e92.png"  width="700" height="350"/> <br/>
    
  <br/>
    
  <img src="https://user-images.githubusercontent.com/103362361/187386154-609a16be-80f0-448a-8033-e97df87c3954.png"  width="500" height="300"/> <br/>
    
  < 예측 결과 이미지 >



### Envs and Requirements
- Colab, Python, MMDetection, Pandas, OpenCV



### Review
-데이터 분석
 - 데이터셋의 이미지와 마스킹 좌표 시각화 
 
 
 <img src="https://user-images.githubusercontent.com/103362361/188362239-3ce1dbd7-856b-44c9-962e-fc9e23897dcd.png"  width="250" height="250"/> <img src="https://user-images.githubusercontent.com/103362361/188362346-9c810636-9c27-438d-99b0-ad0f77bf76ca.png"  width="250" height="250"/>
 - 💡 시각화 해본 결과, 마스킹된 부분들은 이미지만 봤을때 예상한 마스킹보다 훨씬 적었다. 
 - ➡ 데이터의 복잡도가 낮다고 판단, 복잡한 모델을 사용하거나 과한 Augmentation이 적용될 경우 성능이 저하되지 않을까 라는 생각을 하였다. 




### References
- https://github.com/open-mmlab/mmdetection
- https://greeksharifa.github.io/references/2021/09/05/MMDetection02/  
- https://velog.io/@dust_potato/MM-Detection-Config-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0

 






























<!--

## Description

대회 링크 : https://aifactory.space/competition/detail/2067 <br/>

대회 기간 : 2022.07.07(목) 8:00 ~ 2022.08.08(월) 18:00 <br/>

주최 : LG화학 <br/>

주관 : AIFactory <br/>

주제 : 유체상에 떠다니는 입자를 촬영한 화상을 바탕으로 각 입자와 그 형상을 최대한 잘 검출해내는 Instance Segmentation 모델 개발. <br/>

팀 : DeepDream <br/>




<img src="https://user-images.githubusercontent.com/103362361/187381517-1abe006c-6073-4f27-9e55-ae35d388828e.png"  width="500" height="300"/> <br/>

---

## WorkFlow 


각 진행상황에 대해 왜 이렇게 했고 어떻게 생각을 했는지에 대한 원인, 고찰이 들어가야 한다.
왜 워크플로우를 왜 이렇게 잡았는지, 왜 Augmentation을 리스트업 하고 Backbone을 정했는지.



### 1. 데이터  
   
   LG화학에서 제공하는 유체상에 떠다니는 입자를 촬영한 사진. <br/>
   
   train dataset 520장, Test dataset 350장 및 coco dataset 형식의 어노테이션 파일(입자 레이블링 형식에 따라 label_train.json, label(polygon)train.json)  <br/>
   
   객체 카테고리는 1개(Normal) 클래스만 존재, 이미지 해상도는 (Height, Width) = (1024, 1280) 크기 <br/>

   - 데이터 분석
     - 데이터셋의 이미지와 마스킹 좌표 시각화 
     - <img src="https://user-images.githubusercontent.com/103362361/188362239-3ce1dbd7-856b-44c9-962e-fc9e23897dcd.png"  width="250" height="250"/> <img src="https://user-images.githubusercontent.com/103362361/188362346-9c810636-9c27-438d-99b0-ad0f77bf76ca.png"  width="250" height="250"/>
     - 💡 시각화 해본 결과, 마스킹된 부분들은 이미지만 봤을때 예상한 마스킹보다 훨씬 적었다. 
     - ➡ 데이터의 복잡도가 낮다고 판단, 복잡한 모델을 사용하거나 과한 Augmentation이 적용될 경우 성능이 저하되지 않을까 라는 생각을 하였다. 


### 2. 진행 내용
   
   #### 1주차(7/13 ~ 7/17)  
   
   Instance segmentation 공부, MMdetection 라이브러리 사용법 익히기, base-line 돌려보기, EDA(데이터분석) <br/>
   
   MMdetection Reference : [MMDetection tutorial](https://greeksharifa.github.io/references/2021/09/05/MMDetection02/),  [MMDetection Config](https://velog.io/@dust_potato/MM-Detection-Config-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)

   #### 2주차(7/18 ~ 7/24)  
   
   Segmentation model 조원들에게 분배후 제출하여 점수가 높은 모델들 선정 후 model 공부, modeling <br/>
   
   
   
   - 💬 왜 모델을 분배하였는가❔❔
     - MMDetection에는 많은 Instace Segmentation 모델이 존재.
     
     - 데이터 분석후 데이터와 맞지 않다는 모델들을 제외하고도 여러 많은 모델들이 존재하는데 각 모델들을 공부한다음 데이터와 맞다고 생각되는 모델을 정하기엔 시간이 촉박하여 조원들에게 모델들을 분배후 성능을 보고 모델을 선정함.
     
     
     
   - 💬 왜 모델을 먼저 선택하고 Augmentation순으로 Workflow를 잡았는가 ❔❔
     - 데이터를 보고 데이터의 복잡도가 높지 않다고 판단. 먼저 Augmentation을 해주면 데이터의 복잡도가 증가하는데 어떤 증강 기법들을 사용해야 최적의 성능을 내는지 알수없음.
     - 모델마다 복잡도도 다양하기에 먼저 모델을 고정시키고 Augmentation을 다양하게 적용시켜 성능향상을 보는것이 좋을거라 생각하여 모델을 먼저 선택하였다.
     
     
     
     <br/>
     
   
   
   분배모델| 점수
   -------|-------|
   SCNet_r50_fpn_1x(12epoch)_coco  |  0.5861291233  |
   Solov2 |  0.5393581245  |
   Cascade Mask R-CNN_r50_fpn_1x_coco | 0.5850912865 |
   Mask Scoring R-CNN_r50_fpn_1x_coco | 0.5636328897  |
   Hybird Task Cascade=htc_r50_fpn_1x_coco | 0.5548114978  |
   Mask R-CNN_r50_fpn_1x_coco  |  0.5472226479 |
   
 
    
   => 👍 SCNet, Mask R-CNN, Mask Scoring R-CNN, Cascade Mask R-CNN  선정 
   
   ( Mask R-CNN은 다른 모델들의 베이스 모델이기에 같이 공부하려고 선정! )

   <br/>

   #### 3주차(7/25 ~ 7/31)  
   
   전처리 
   - Transform(이미지중 일부를 변형하여 학습에 사용)  
      
   - Augmentation(이미지 증강) 리스트업하여 각각 어떤 기법을 사용할 것인지 분배하고 성능확인 및 어떤걸 쓸지 선정.
   
   - backbone 분배후 성능확인 및 backbone 선정 
   
   Backbone |
   -------|
   ResNet stricks back, EfficientNet, ConvNeXt   |  
   HRNet, Generalized Attention  |  
   GCNet, Res2Net |  
   PVT, PVTv2, Swin  |  
   RegNet, ResNest | 
   
   => 👍 ResNext, ResNet strikes back을 최종 Backbone으로 선정.
   
   
   <br/>
   
   
   - Online Data augmentation list-up(Transform)  
     - shear, rotate, resize, flip, equalize, brightness, contrast, minIoUrandomcrop, Albumentation
     - 동일 조건으로 고정. - cascade_rcnn_x101_32x4d_fpn_1x_coco, 12epoch, IOU threshold(0.3 ~ 0.6)
   
   
   Augmentation | 점수 |
   -------|-------|
   all augmentation   |  0.5452023496  |
   resize : 1024,1024 |  0.6041639844  |
   resize : 1280,1024 |  0.6084128911  |
   resize : 1333,800  |  0.6065363398  |
   equalize, brightness, contrast |  0.6044010023  |
   albumentation(Equalize, Brightness, contrast)  |  0.6042227322  |
   miniourandomcrop | 0.6023441395 |
     
   <br/>



   #### 4주차(8/1 ~ 8/8)  
   
   Optimizer 분배 후 선정, 성능향상을 위한 하이퍼파라미터 조정. 
   
   Optimizer |
   -------|
   RMSprop, Rprop   |  
   ASGD, LBFGS  |  
   Adadelta, Adagrad |  
   NAdam, Radam  |  
   SparseAdam, Adamax | 
    
   => 👍 Adadelta를 최종 Opitmizer로 선정
   
   <br/><br/>
   
   
  
  
  
  
  ### 3. 결과 : 62팀 중 7등 달성!
    
    
    
  <img src="https://user-images.githubusercontent.com/103362361/187385788-913ff59d-cc4a-4d4a-bebc-456c99575e92.png"  width="600" height="400"/> <br/>
    
  <br/>
    
  <img src="https://user-images.githubusercontent.com/103362361/187386154-609a16be-80f0-448a-8033-e97df87c3954.png"  width="600" height="400"/> <br/>
    
  < 예측 결과 이미지 >
    

   
---

## Usage Library & Baseline

Usage Library : [MMdetection](https://github.com/open-mmlab/mmdetection)

Base line : Mask R-CNN (대회측에서 baseline code 제공)

-->


<!-- 모델의 성능도 중요하지만 인퍼런스속도가 얼마나 나오냐, 걸리는 시간에 대한 리뷰가 있으면 좋겠다..
아쉬운점 : 우리의 상세한 과정도 나쁘지 않지만 체계적으로 들리진 않았다. = 해볼수 있는걸 다해봤는데 그중 좋을걸 골라서 해봤다. 논리적인 체계가 있어야. 근거가 정확하게 있어서 이거를 이렇게 했고 인사이트가 도출되어서, 나온 인사이트를 보여줘야. 이게 잘 보이지 않으니 다른 문제를 풀면 잘할까? 라는 의문이 듬.
우리가 어떤 역량을 가지고 있는지를 보여줘야하는데 7등했다말고는 보이는게 없음.
논문을 어느정도 리뷰하는내용은 굳이 필요하진 않음. 
-->





