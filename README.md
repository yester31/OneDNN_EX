# OneDNN_EX

## Enviroments
- Windows 10 laptop
- CPU i7-11375H
- Opencv 3.4.5
- OneDNN 2.50 (2021.4)


## OneDNN Simple Classification model
- resnet18 Model
- Function abstraction in an easy-to-use form  
- Primitive function validation 
- Weight loader 
- Data loader 
- Image preprocess 
- Modeling 
- Match all results with PyTorch 
- Optimization 
- Performace evaluation
- 1) Specific data type version  -> 6765 [ms] (resnet18_v1.cpp)
- 2) Any data type version  -> 2466 [ms] (resnet18_v2.cpp)
- PyTorch (cpu) -> 2759 [ms]

## Custom Primitive using DPC++(preparing)
-


## Reference
- oneDNN github : <https://github.com/oneapi-src/oneDNN#installation>
- onednn v2.50 documentation :<https://oneapi-src.github.io/oneDNN/>
- download : <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html#gs.i239i5>
