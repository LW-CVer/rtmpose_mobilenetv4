# rtmpose_mobilenetv4
将rtmpose的backbone替换为mobilenetv4

## 安装,
### 官方源码安装方式
https://mmpose.readthedocs.io/en/latest/installation.html

- conda create --name rtmpose python=3.8 -y
- conda activate rtmpose
- conda install pytorch torchvision -c pytorch

- pip install -U openmim
- mim install mmengine
- mim install "mmcv>=2.0.1"

### 进入代码根目录
- cd mmpose
- pip install -r requirements.txt
- pip install -v -e .
- mim install "mmpose>=1.1.0"

### 验证是否安装成功
- python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file vis_results.jpg \
    --draw-heatmap

### 训练模型
python tools/train.py configs\body_2d_keypoint\rtmpose\coco\mv4_rtmpose-m_8xb256-420e_coco-256x192.py