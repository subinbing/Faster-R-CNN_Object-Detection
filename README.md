# Faster-R-CNN_Object-Detection

1. data 폴더 추가

2. data 구조

data
- class1
  - img
    - image1.jpg
    - image2.jpg
      
      ...
    - imagen.jpg
  - json
      - json1.json
      - json2.json
        
        ...
      - jsonn.json
- class2
  - img
    - image1.jpg
    - image2.jpg
      
      ...
    - imagen.jpg
  - json
      - json1.json
      - json2.json
        
        ...
      - jsonn.json
- class3
  - img
    - image1.jpg
    - image2.jpg
      
      ...
    - imagen.jpg
  - json
      - json1.json
      - json2.json
        
        ...
      - jsonn.json
        
 ...

 
- classn


3. data에 형식에 맞게 dataset_fasterrcnn.py 수정


4. Anaconda 가상환경

conda create -n fasterrcnn python=3.8
conda activate fasterrcnn

5. train

python train_fasterrcnn_resnet50.py

6. test
   
python test_fasterrcnn_resnet50.py
