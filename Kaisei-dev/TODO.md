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
                     2784 characters in annotation, case sensitive.
- Geometry Notation = RBOX
- Detection Granularity = Text-line
- Detection Output Fashion = Per-pixel

## RoadMap
#### Feature Extractor: ResNet & DeConv (11 Apr) - Done & Executable, 11 Apr
#### Det Branch: EAST Trans, visualize layers and CUDA Enable Check (12-13 Apr)
PyTorch 0.3.1 no longer support old GPUs like Quadro K4200 (3.0).
#### DataLoader & Helper Func (13 Apr + 4)
#### Rec Branch: CRNN Trans (16 Apr + 1)
#### RoI Op, Loss Design and Optimizer(19 Apr)
#### Training, Augmentation(Recognition), OHEM etc. (19 Apr ~)
#### OPTIONAL: add crayon


## Out-of-Box Annotation Files
ts-0548
tr-0170
tr-0874 
