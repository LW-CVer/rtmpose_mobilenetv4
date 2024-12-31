import torch 

ckpt=torch.load('work_dirs/rtmpose-m_8xb256-420e_coco-256x192/best_coco_AP_epoch_410.pth')
for i in ckpt.keys():
    print(i)
torch.save(ckpt['state_dict'], 'work_dirs/rtmpose-m_8xb256-420e_coco-256x192/best_coco_AP_epoch_410.onnx')