# ZFNet
ZFNet implemented in pytorch


![image](https://user-images.githubusercontent.com/55650445/125544264-5c2cdacd-c859-4da3-8fb8-05a231648d2d.png)


### C1
nn.Conv2d(3, 96, k=7, stride=2, padding=1)
nn.ReLU()
nn.MaxPool2d(k=3, stride=2, padding=1)
nn.LocalResponseNorm(5)  

### C2
nn.Conv2d(96, 256, k=5, stride=2)
nn.ReLU()
nn.MaxPool2d(k=3, stride=2, padding=1)
nn.LocalResponseNorm(5)  

### C3
nn.Conv2d(256, 384, k=3, stride=1, padding=1)
nn.ReLU()  

### C4
nn.Conv2d(384, 384, k=3, stride=1, padding=1)
nn.ReLU()  

### C5
nn.Conv2d(384, 256, k=3, stride=1, padding=1)
nn.ReLU()
nn.MaxPool2d(k=3, stride=2)  

### F1
nn.Linear(9216, 4096)
nn.Dropout()  

### F2
nn.Linear(4096, num_classes)
nn.Dropout()
