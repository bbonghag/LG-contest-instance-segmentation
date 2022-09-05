## LG화학 입자 형태 분석 모델 개발 해커톤 - DeepDream Rank7🏅🎖
### 입자 검출 정보를 기반으로 입자들의 형태 변화를 계량적 지표로 산출 가능한 모델 개발

---

## Description

대회 링크 : https://aifactory.space/competition/detail/2067 <br/>

대회 기간 : 2022.07.07(목) 8:00 ~ 2022.08.08(월) 18:00 <br/>

주최 : LG화학 <br/>

주관 : AIFactory <br/>

주제 : 유체상에 떠다니는 입자를 촬영한 화상을 바탕으로 각 입자와 그 형상을 최대한 잘 검출해내는 Instance Segmentation 모델 개발. <br/>

팀 : DeepDream(조장 : 김수현, 조원 : [김현나](https://github.com/hna12), 이x정, 소x희, 이봉학) <br/>




<img src="https://user-images.githubusercontent.com/103362361/187381517-1abe006c-6073-4f27-9e55-ae35d388828e.png"  width="500" height="300"/> <br/>

---

## WorkFlow 

<!-- 
각 진행상황에 대해 왜 이렇게 했고 어떻게 생각을 했는지에 대한 원인, 고찰이 들어가야 한다.
왜 워크플로우를 왜 이렇게 잡았는지, 왜 Augmentation을 리스트업 하고 Backbone을 정했는지.
-->

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
     - MMDetection에는 많은 Instace Segmentation 모델이 존재합니다
     
     - 데이터 분석후 데이터와 맞지 않다는 모델들을 제외하고도 여러 많은 모델들이 존재하는데 각 모델들을 공부한다음 데이터와 맞다고 생각되는 모델을 정하기엔 시간이 촉박하여 조원들에게 모델들을 분배후 성능을 보고 모델을 선정하기로 하였습니다
     
     <br/>
     
   
   
   조원 | 분배모델| 점수
   -----|-------|-------|
   현나  |  SCNet_r50_fpn_1x(12epoch)_coco  |  0.5861291233  |
   수현  |  Solov2 |  0.5393581245  |
   소x   |  Cascade Mask R-CNN_r50_fpn_1x_coco | 0.5850912865 |
   소x   |  Mask Scoring R-CNN_r50_fpn_1x_coco | 0.5636328897  |
   봉학  |  Hybird Task Cascade=htc_r50_fpn_1x_coco | 0.5548114978  |
   국x   |  Mask R-CNN_r50_fpn_1x_coco  |  0.5472226479 |
   
 
    
   => 👍 SCNet, Mask R-CNN, Mask Scoring R-CNN, Cascade Mask R-CNN  선정 
   
   ( Mask R-CNN은 다른 모델들의 베이스 모델이기에 같이 공부하려고 선정! )

   <br/>

   #### 3주차(7/25 ~ 7/31)  
   
   전처리 
   - Transform(이미지중 일부를 변형하여 학습에 사용)  
      
   - Augmentation(이미지 증강) 리스트업하여 각각 어떤 기법을 사용할 것인지 분배하고 성능확인 및 어떤걸 쓸지 선정.
   
   - backbone 분배후 성능확인 및 backbone 선정 <br/>
   
   
   - Online Data augmentation list-up(Transform)  
     - shear, rotate, resize, flip, equalize, brightness, contrast, minIoUrandomcrop, Albumentation
     - 동일 조건으로 고정. - cascade_rcnn_x101_32x4d_fpn_1x_coco, 12epoch, IOU threshold(0.3 ~ 0.6)
   
   
   조원 | Augmentation | 점수 |
   -----|-------|-------|
   수현 | all augmentation   |  0.5452023496  |
   국x  | resize : 1024,1024 |  0.6041639844  |
   국x  | resize : 1280,1024 |  0.6084128911  |
   국x  | resize : 1333,800  |  0.6065363398  |
   소x  | equalize, brightness, contrast |  0.6044010023  |
   봉학 | albumentation(Equalize, Brightness, contrast)  |  0.6042227322  |
   현나 | miniourandomcrop | 0.6023441395 |
     
   <br/>



   #### 4주차(8/1 ~ 8/8)  
   
   Optimizer, Lr-scheduler 분배 후 선정, 성능향상을 위한 하이퍼파라미터 조정. <br/><br/>
   
  
  
  
  
  
   
 3. 결과 : 62팀 중 7등 달성!
    
    
    
    <img src="https://user-images.githubusercontent.com/103362361/187385788-913ff59d-cc4a-4d4a-bebc-456c99575e92.png"  width="600" height="400"/> <br/>
    
    <br/>
    
    <img src="https://user-images.githubusercontent.com/103362361/187386154-609a16be-80f0-448a-8033-e97df87c3954.png"  width="600" height="400"/> <br/>
    
       < 예측 결과 이미지 >
    

   
---

## Usage Library & Baseline

Usage Library : [MMdetection](https://github.com/open-mmlab/mmdetection)

Base line : Mask R-CNN (대회측에서 baseline code 제공)

---




