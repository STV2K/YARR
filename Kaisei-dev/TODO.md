# KAISEI Detection Branch Developing Note

Using PyTorch.

## Transplant
### EAST [Code](https://github.com/argman/east)
### CRNN [Code](https://github.com/bgshih/crnn)

## Invent
### Experiment Idea on Score Branch

## Specs
- STV2K Annotation = Quadrilateral, Text-line, GBK Encoding,
                     1215 Tr, 853 Ts. See [paper]() for detail.
- Geometry Notation = RBOX
- Detection Granularity = Text-line
- Detection Output Fashion = Per-pixel

## RoadMap
#### Feature Extractor: ResNet & DeConv (11 Apr) - Done & Executable, 11 Apr
#### Det Branch: EAST Trans, visualize layers and CUDA Enable Check (12-13 Apr)
Cuda: `Variable.cuda()` and `model.cuda()`.  
However PyTorch 0.3.1 no longer support old GPUs like Quadro K4200 (3.0).
#### DataLoader & Helper Func (13 Apr + 1)
Start Detection Branch Experiment  
Confirm Rec Progress - Dict...  
#### Rec Branch: CRNN Trans (16 Apr)
#### RoI Op, Loss Design and Optimizer(18 Apr)
#### Training, Augmentation, OHEM etc. (18 Apr ~)
#### OPTIONAL: add crayon