"""PE34 homework: Rock/Paper/Scissors transfer learning.
Run in Google Colab/Kaggle with GPU. Dataset: drgfreeman/rockpaperscissors.
"""
from pathlib import Path
import os, random, shutil, time, copy
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, torchvision
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler

SEED=42; CLASSES=['rock','paper','scissors']; EXT={'.jpg','.jpeg','.png','.bmp','.webp'}
RAW=Path('/content/data/raw'); OUT=Path('/content/data/rps_prepared')
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# In Colab: upload kaggle.json to /root/.kaggle/kaggle.json before running this script.
def download_dataset():
    RAW.mkdir(parents=True, exist_ok=True)
    os.system('pip -q install kaggle')
    code=os.system('kaggle datasets download -d drgfreeman/rockpaperscissors -p /content/data/raw --unzip')
    if code!=0: raise RuntimeError('Kaggle download failed: check kaggle.json')

def find_root(base=RAW):
    for root, dirs, _ in os.walk(base):
        if all(c in {d.lower() for d in dirs} for c in CLASSES): return Path(root)
    raise FileNotFoundError('Class folders rock/paper/scissors not found')

def valid_img(p):
    try:
        Image.open(p).verify(); return True
    except Exception: return False

def prepare_dataset(src, out=OUT):
    if out.exists(): shutil.rmtree(out)
    rows=[]
    for split in ['train','val','test']:
        for c in CLASSES: (out/split/c).mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        files=[p for p in sorted((src/c).iterdir()) if p.suffix.lower() in EXT and valid_img(p)]
        train,tmp=train_test_split(files, train_size=.70, random_state=SEED, shuffle=True)
        val,test=train_test_split(tmp, train_size=.50, random_state=SEED, shuffle=True)
        for split, items in {'train':train,'val':val,'test':test}.items():
            for f in items: shutil.copy2(f, out/split/c/f.name)
            rows.append({'class':c,'split':split,'count':len(items)})
    return pd.DataFrame(rows)

def loaders(data=OUT, batch=32):
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    tr=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.7,1)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(12),transforms.ColorJitter(.15,.15,.15),transforms.ToTensor(),norm])
    ev=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
    ds={s:datasets.ImageFolder(data/s, tr if s=='train' else ev) for s in ['train','val','test']}
    dl={s:torch.utils.data.DataLoader(ds[s],batch_size=batch,shuffle=(s=='train'),num_workers=2) for s in ds}
    return ds, dl, {s:len(ds[s]) for s in ds}, ds['train'].classes

def show_batch(dl, names, n=16):
    x,y=next(iter(dl)); mean=torch.tensor([.485,.456,.406]).view(3,1,1); std=torch.tensor([.229,.224,.225]).view(3,1,1)
    imgs=torch.stack([torch.clamp(x[i].cpu()*std+mean,0,1) for i in range(min(n,len(x)))])
    grid=torchvision.utils.make_grid(imgs,nrow=4); plt.figure(figsize=(8,8)); plt.imshow(grid.permute(1,2,0)); plt.axis('off'); plt.title(' | '.join(names[y[i]] for i in range(min(n,len(y))))); plt.show()

def train_model(model, dl, sizes, crit, opt, sch=None, epochs=5, name='model'):
    best=copy.deepcopy(model.state_dict()); best_acc=0; start=time.time(); hist={'epoch':[],'train_acc':[],'val_acc':[],'train_loss':[],'val_loss':[]}
    for e in range(epochs):
        print(f'\n{name}: epoch {e+1}/{epochs}')
        for phase in ['train','val']:
            model.train() if phase=='train' else model.eval(); loss_sum=0; ok=0
            for inp, lab in tqdm(dl[phase], desc=phase):
                inp,lab=inp.to(DEVICE),lab.to(DEVICE); opt.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    out=model(inp); loss=crit(out,lab); pred=out.argmax(1)
                    if phase=='train': loss.backward(); opt.step()
                loss_sum+=loss.item()*inp.size(0); ok+=(pred==lab).sum().item()
            if phase=='train' and sch: sch.step()
            loss=loss_sum/sizes[phase]; acc=ok/sizes[phase]; print(phase, round(loss,4), round(acc,4))
            hist[f'{phase}_loss'].append(loss); hist[f'{phase}_acc'].append(acc)
            if phase=='val': hist['epoch'].append(e+1); best_acc, best = (acc,copy.deepcopy(model.state_dict())) if acc>best_acc else (best_acc,best)
    model.load_state_dict(best); return model, pd.DataFrame(hist), best_acc, time.time()-start

def build_scratch(n):
    m=models.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,n); return m.to(DEVICE)

def build_transfer(n):
    m=models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad=False
    m.fc=nn.Linear(m.fc.in_features,n); return m.to(DEVICE)

def evaluate(model, dl, names, title):
    model.eval(); yt=[]; yp=[]
    with torch.no_grad():
        for x,y in tqdm(dl, desc=title):
            p=model(x.to(DEVICE)).argmax(1).cpu().numpy(); yp+=p.tolist(); yt+=y.numpy().tolist()
    acc=accuracy_score(yt,yp); print(title,'test_acc=',round(acc,4)); print(classification_report(yt,yp,target_names=names))
    cm=confusion_matrix(yt,yp); plt.imshow(cm); plt.title(title); plt.colorbar(); plt.xticks(range(len(names)),names); plt.yticks(range(len(names)),names); plt.show(); return acc

def main():
    download_dataset(); root=find_root(); print(prepare_dataset(root).pivot(index='class',columns='split',values='count'))
    ds,dl,sizes,names=loaders(); show_batch(dl['train'],names); crit=nn.CrossEntropyLoss()
    scratch=build_scratch(len(names)); opt=optim.SGD(scratch.parameters(),lr=.001,momentum=.9); sch=lr_scheduler.StepLR(opt,3,.1)
    scratch,h1,v1,t1=train_model(scratch,dl,sizes,crit,opt,sch,5,'scratch')
    transfer=build_transfer(len(names)); opt=optim.SGD(transfer.fc.parameters(),lr=.003,momentum=.9); sch=lr_scheduler.StepLR(opt,3,.1)
    transfer,h2,v2,t2=train_model(transfer,dl,sizes,crit,opt,sch,5,'transfer')
    for n,p in transfer.named_parameters(): p.requires_grad=n.startswith('layer4') or n.startswith('fc')
    opt=optim.SGD(filter(lambda p:p.requires_grad,transfer.parameters()),lr=.0005,momentum=.9)
    transfer,h3,v3,t3=train_model(transfer,dl,sizes,crit,opt,None,3,'fine_tune_layer4')
    a1=evaluate(scratch,dl['test'],names,'scratch'); a2=evaluate(transfer,dl['test'],names,'transfer_fine_tuned')
    print(pd.DataFrame([{'model':'scratch','best_val_acc':v1,'test_acc':a1,'time_min':t1/60},{'model':'transfer_fine_tuned','best_val_acc':max(v2,v3),'test_acc':a2,'time_min':(t2+t3)/60}]))
if __name__=='__main__': main()
