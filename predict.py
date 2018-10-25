import argparse
import torch
import numpy as np
import json
from PIL import Image
from torchvision import datasets, transforms, models

data_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
parser = argparse.ArgumentParser('Image Classifier')
parser.add_argument('path', action='store',help='image path')
parser.add_argument('checkpoint', action='store',help='saved model')
parser.add_argument('--top_k', action='store',dest='top_k', type=int, help='number of top classes')
parser.add_argument('--category_names', action='store',dest='cat_names',help='path to classes names dictionary')
parser.add_argument('--gpu', action='store_true',dest='gpu',default=False,help='add this argument to use gpu')
params=vars(parser.parse_args())
path, checkpoint, top_k, cat_names, gpu=params['path'], params['checkpoint'], params['top_k'], params['cat_names'], params['gpu']
device='cpu'
if gpu:
    device='cuda'
if cat_names!=None:
    with open(cat_names, 'r') as f:
        names_dict = json.load(f)
def process_image(image):
    image=data_transforms(image).view(1, 3, 224, 224)
    return image

def class_name(c):
    if cat_names==None:
        return 'class '+str(c)
    else:
        return names_dict[c]

model=torch.load(checkpoint)
model.eval()
model.to(device)
img=process_image(Image.open(path))
idx_to_class={}
for key, value in model.class_to_idx.items():
    idx_to_class[value] = key
with torch.no_grad():
    out=model(img).data
if top_k==None:
    prob, predicted = torch.max(out, 1)
    prob=(torch.exp(prob)).item()
    predicted=predicted.item()
    print('This is '+class_name(idx_to_class[predicted])+' with probability '+str(round(prob*100,2))+'%')
else:
    probs, indices=out.topk(top_k)
    indices=np.asarray(indices).tolist()[0]
    probs=np.asarray(torch.exp(probs)).tolist()[0]
    print('Top '+str(top_k)+' classes:')
    for i in range(len(indices)):
        print(class_name(idx_to_class[(indices[i])])+' with probability '+str(round(probs[i]*100,2))+'%')
               
        
    


