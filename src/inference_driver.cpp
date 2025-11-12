/*
 * Filename: inference_driver.cpp
 *
 * @Author: GeonhaPark
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 07/23/25
 * @Original Work: Based on minimal-litert-c repository (https://github.com/SNU-RTOS/minimal-litert-c)
 * @Modified by: Namcheol Lee, Taehyun Kim on 10/16/25
 * @Contact: nclee@redwood.snu.ac.kr
 *
 * @Description: Inference driver for executing a LiteRT model with delegate
 *
 */

#include <iostream>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/delegates/gpu/delegate.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "util.hpp"

/* ============ Function Naming Convention of LiteRT ============
 * Public C++ class methods: UpperCamelCase (e.g., BuildFromFile)
 * Internal helpers: snake_case (e.g., typed_input_tensor)
 * ============================================================ */

int main(int argc, char *argv[]) {
    /* Receive arguments */
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] 
            << "<model_path> <gpu_usage> <class_labels_path> <image_path 1> "
            << "[image_path 2 ... image_path N]"
            << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];

    bool gpu_usage = false; // If true, GPU delegate is applied
    const std::string gpu_usage_str = argv[2];
    if (gpu_usage_str == "true") {
        gpu_usage = true;
    }
    
    // Load class label mapping, used for postprocessing
    const std::string class_labels_path = argv[3];
    auto class_labels_map = util::load_class_labels(class_labels_path.c_str());

    std::vector<std::string> image_paths;   // List of input image paths
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        image_paths.push_back(arg);    
    }
    
    /* Load model */
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }


    // ====================================
    
    /* Build interpreter */
    // 1. Create an OpResolver
    // 2. Create an interpreter builder
    // 3. Build an interpreter using the interpreter builder
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to initialize interpreter" << std::endl;
        return 1;
    }
    


    /* Apply either XNNPACK delegate or GPU delegate */
    // 1. Create a XNNPACK delegate
    // 2. Create a GPU delegate 
    // 3. Apply either XNNPACK or GPU delegate
    // 4. Delete the unsed delegate
    TfLiteDelegate* xnn_delegate = TfLiteXNNPackDelegateCreate(nullptr);
    TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(nullptr);

    if (gpu_usage) {
        if (interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk) {
            // Delete unused delegate
            if (xnn_delegate) {
                TfLiteXNNPackDelegateDelete(xnn_delegate);
            }
        } else {
            std::cerr << "Failed to Apply GPU Delegate" << std::endl;
        }
    } else {
        if (interpreter->ModifyGraphWithDelegate(xnn_delegate) == kTfLiteOk) {
            // Delete unused delegate
            if (gpu_delegate) {
                TfLiteGpuDelegateV2Delete(gpu_delegate);
            }
        } else {
            std::cerr << "Failed to Apply XNNPACK Delegate" << std::endl;
        }
    }

    /* Allocate Tensors */
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to Allocate Tensors" << std::endl;
        return 1;
    }
    
    
    // Starting inference
    util::timer_start("Total Latency");

    int count = 0;
    
    do {
        std::string e2e_label = "E2E" + std::to_string(count);
        std::string preprocess_label = "Preprocessing" + std::to_string(count);
        std::string inference_label = "Inference" + std::to_string(count);
        std::string postprocess_label = "Postprocessing" + std::to_string(count);

        util::timer_start(e2e_label);
        util::timer_start(preprocess_label);
        /* Preprocessing */
        // Load input image
        cv::Mat image = cv::imread(image_paths[count]);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_paths[count] << "\n";
            continue;
        }

        // Preprocess input data
        // 1. call util::preprocess_image to flatten a image
        // 2. Copy the preprocessed data to input tensor
        cv::Mat preprocessed_image = util::preprocess_image(image, 28, 28);

        float* input_tensor = interpreter->typed_input_tensor<float>(0);
        std::memcpy(input_tensor, preprocessed_image.ptr<float>(),
                    preprocessed_image.total() * preprocessed_image.elemSize());


        // ====================================
        util::timer_stop(preprocess_label);

        util::timer_start(inference_label);
        /* Inference */
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke the interpreter" << std::endl;
            return 1;
        }
        


        // ====================================
        util::timer_stop(inference_label);

        util::timer_start(postprocess_label);
        /* PostProcessing */
        // 1. Get output tensor
        // 2. Apply softmax to get probabilities        
        float *output_tensor = interpreter->typed_output_tensor<float>(0);
        int num_classes = 10;
        std::vector<float> logits(num_classes);
        std::memcpy(logits.data(), output_tensor, sizeof(float) * num_classes);

        std::vector<float> probs(num_classes);
        util::softmax(logits.data(), probs, num_classes);
        
        
        // ====================================
        // Print Top-3 predictions every 10 iterations
        if ((count + 1) % 10 == 0) {
            std::cout << "\n[INFO] Top 3 predictions for image index " << count << ":" 
            << std::endl;
            auto top_k_indices = util::get_topK_indices(probs, 3);
            for (int idx : top_k_indices) {
                std::string label = 
                    class_labels_map.count(idx) ? class_labels_map[idx] : "unknown";
                std::cout << "- Class " << idx << " (" << label << "): " 
                    << probs[idx] << std::endl;
            }
        }

        util::timer_stop(postprocess_label);
        util::timer_stop(e2e_label);

        ++count; // Processed image count
    } while (count < image_paths.size());
    util::timer_stop("Total Latency");

    /* Print average E2E latency and throughput */
    util::print_average_latency("E2E");
    util::print_average_latency("Preprocessing");
    util::print_average_latency("Inference");
    util::print_average_latency("Postprocessing");
    util::print_throughput("Total Latency", image_paths.size());    

    return 0;
}
