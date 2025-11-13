# On Device AI Bootcamp Source 

## Demo

0. Env setup

Run `sudo ./scripts/install_prerequisites.sh` first.

Then run `./setup.sh`.

1. Build Model

Run `python ./python/train.py`

After running `python ./python/train.py`, you will get a model file `simple_classifier.onnx` in models directory.

2. Convert onnx model to tflite model

Run `python ./python/convert.py --onnx-path models/simple_classifier.onnx --output-dir models/litert`

3. Build Inference Driver

Run `make inference`

4. Run Model

Run `./run_inference_driver.sh`

5. Check Instrumental Harness

Run `make harness`

Then run `./bin/instrumentation_harness ./models/litert/simple_classifier_float32.tflite`

## Docs

Check [ppt file](https://github.com/yyc9920/OnDeviceAI-Bootcamp/tree/main/ppt/OnDeviceAI_Knowledge_share.pptx)
