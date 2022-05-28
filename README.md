# OneDNN_EX

## Enviroments
- Windows 10 laptop
- CPU i7-11375H
- Opencv 3.4.5
- OneDNN 2.50 (2021.4)


## OneDNN Simple Classification model (cpu, F32)
- resnet18 Model
- Function abstraction in an easy-to-use form  
- Primitive function validation 
- Weight loader 
- Data loader 
- Image preprocess 
- Modeling 
- Match all results with PyTorch 
- Optimization 
- Performace evaluation(Execution time of 100 iteration for one 224x224x3 image)

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Pytorch</strong></td><td><strong>OneDNN</strong></td><td><strong>OneDNN</strong></td><td><strong>OneDNN</strong></td>
		</tr>
		<tr>
			<td>Description</td><td>general</td><td>Specific data type</td><td>Any data type</td><td>Any data type + fused post ops</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>27.59 ms</td>
			<td>67.65 ms </td>
			<td>24.40 ms</td>
			<td>22.81 ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td>36 fps</td>
			<td>15 fps</td>
			<td>41 fps</td>
			<td>44 fps</td>
		</tr>
		<tr>
			<td>File</td>
			<td>Resnet18_py/inference.py</td>
			<td>resnet18_v1.cpp</td>
			<td>resnet18_v2.cpp</td>
			<td>resnet18_v3.cpp</td>
		</tr>
	</tbody>
</table>


## Reference
- oneDNN github : <https://github.com/oneapi-src/oneDNN#installation>
- onednn v2.50 documentation :<https://oneapi-src.github.io/oneDNN/>
- download : <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html#gs.i239i5>
