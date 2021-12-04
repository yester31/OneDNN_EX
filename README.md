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
- Performace evaluation(cpu, F32)
- 1) Reference PyTorch (Resnet18_py/inference.py)
        - 2759 [ms]
- 2) Specific data type version (resnet18_v1.cpp)  
        - 6765 [ms] 
- 3) Any data type version (resnet18_v2.cpp)
        - 2440 [ms] 
- 4) Any data type + fused post ops version (resnet18_v3.cpp)
        - 2281 [ms] 


## Custom Primitive using DPC++(preparing)
-


## Reference
- oneDNN github : <https://github.com/oneapi-src/oneDNN#installation>
- onednn v2.50 documentation :<https://oneapi-src.github.io/oneDNN/>
- download : <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html#gs.i239i5>
