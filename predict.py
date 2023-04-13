import torch
import numpy as np
import cv2
import os
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.profiler import profile, record_function, ProfilerActivity

saved_model = "saved_model"  # Output directory of the save the model
filename = "IMG_2558.JPG"    # Image filename
img_path = "cards_dataset/validation/" + filename

with open('labels.txt', 'r') as f:
	string = f.read()
	labels_dict = eval(string)

def get_model(num_classes):

	# Load an pre-trained object detectin model (in this case faster-rcnn)
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

	# Number of input features
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# Replace the pre-trained head with a new head
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model

image = cv2.imread(img_path)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = torchvision.transforms.ToTensor()(img)

loaded_model = get_model(6)
loaded_model.load_state_dict(torch.load(os.path.join(saved_model, 'model'), map_location = 'cpu'))

loaded_model.eval()


# with torch.autograd.profiler.profile(
#         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         profile_memory=True,  # Enable memory profiling
#         record_shapes=True,
#         use_cuda=True) as prof:
    # Run inference on the original model
	# with torch.no_grad():
	# with record_function("original_model_inference"):
with torch.autograd.profiler.profile(with_stack=True) as prof:
	with torch.no_grad():
        # output = model(input_tensor)
		prediction = loaded_model([img])

    # with record_function("original_model_inference"):
    #     output = model(input_tensor)
        
    # # Run inference on the quantized model
    # with record_function("quantized_model_inference"):
    #     quantized_output = quantized_model(input_tensor)
memory_allocated_before_original_model = prof.total_average().torch.cuda.memory_allocated()()
print(memory_allocated_before_original_model)
# Print memory usage for original and quantized model
# print("Original Model Memory Usage: {} bytes".format(prof.total_average().cuda_memory_usage["allocated_bytes.all.allocated"]))
# print("Quantized Model Memory Usage: {} bytes".format(prof.total_average().cuda_memory_usage["allocated_bytes.all.allocated"]))

for element in range(len(prediction[0]['boxes'])):
	x, y, w, h = prediction[0]['boxes'][element].numpy().astype(int)
	score = np.round(prediction[0]['scores'][element].numpy(), decimals = 3)
	label_index = prediction[0]['labels'][element].numpy()
	label = labels_dict[int(label_index)]
	
	if score > 0.45:
		cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 2)
		text = str(label) + " " + str(score)
		cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					 (255, 255, 255), 1)
					 


cv2.imshow("Predictions", image)
cv2.waitKey(0)