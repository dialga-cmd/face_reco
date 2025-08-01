// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <memory>
#include "default_providers.h"
#include "providers.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#ifdef USE_COREML
#include "core/providers/coreml/coreml_provider_factory.h"
#endif
#ifdef USE_CUDA
#include <core/providers/cuda/cuda_provider_options.h>
#endif
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/session_options.h"

namespace onnxruntime {

namespace test {

std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  return CPUProviderFactoryCreator::Create(enable_arena)->CreateProvider();
}

std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider() {
#ifdef USE_TENSORRT
  OrtTensorRTProviderOptions params{
      0,
      0,
      nullptr,
      1000,
      1,
      1 << 30,
      0,
      0,
      nullptr,
      0,
      0,
      0,
      0,
      0,
      nullptr,
      0,
      nullptr,
      0};
  if (auto factory = TensorrtProviderFactoryCreator::Create(&params))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultNvTensorRTRTXExecutionProvider() {
#ifdef USE_NV
  if (auto factory = NvProviderFactoryCreator::Create(0))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptions* params) {
#ifdef USE_TENSORRT
  if (auto factory = TensorrtProviderFactoryCreator::Create(params))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptionsV2* params) {
#ifdef USE_TENSORRT
  if (auto factory = TensorrtProviderFactoryCreator::Create(params))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultMIGraphXExecutionProvider() {
#ifdef USE_MIGRAPHX
  OrtMIGraphXProviderOptions params{
      0,
      0,
      0,
      0,
      0,
      nullptr,
      1,
      "./compiled_model.mxr",
      1,
      "./compiled_model.mxr",
      1,
      SIZE_MAX,
      0};
  return MIGraphXProviderFactoryCreator::Create(&params)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> MIGraphXExecutionProviderWithOptions(const OrtMIGraphXProviderOptions* params) {
#ifdef USE_MIGRAPHX
  if (auto factory = MIGraphXProviderFactoryCreator::Create(params))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> OpenVINOExecutionProviderWithOptions(const ProviderOptions* params,
                                                                         const SessionOptions* session_options) {
#ifdef USE_OPENVINO
  return OpenVINOProviderFactoryCreator::Create(params, session_options)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
  ORT_UNUSED_PARAMETER(session_options);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider() {
#ifdef USE_OPENVINO
  ProviderOptions provider_options_map;
  SessionOptions session_options;
  return OpenVINOProviderFactoryCreator::Create(&provider_options_map, &session_options)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider() {
#ifdef USE_CUDA
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  if (auto factory = CudaProviderFactoryCreator::Create(&provider_options))
    return factory->CreateProvider();
#endif
  return nullptr;
}

#ifdef ENABLE_CUDA_NHWC_OPS
std::unique_ptr<IExecutionProvider> DefaultCudaNHWCExecutionProvider() {
#if defined(USE_CUDA)
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  provider_options.prefer_nhwc = true;
  if (auto factory = CudaProviderFactoryCreator::Create(&provider_options))
    return factory->CreateProvider();
#endif
  return nullptr;
}
#endif

std::unique_ptr<IExecutionProvider> CudaExecutionProviderWithOptions(const OrtCUDAProviderOptionsV2* provider_options) {
#ifdef USE_CUDA
  if (auto factory = CudaProviderFactoryCreator::Create(provider_options))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(provider_options);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider() {
#ifdef USE_DNNL
  OrtDnnlProviderOptions dnnl_options;
  dnnl_options.use_arena = 1;
  dnnl_options.threadpool_args = nullptr;
  if (auto factory = DnnlProviderFactoryCreator::Create(&dnnl_options))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DnnlExecutionProviderWithOptions(const OrtDnnlProviderOptions* provider_options) {
#ifdef USE_DNNL
  if (auto factory = DnnlProviderFactoryCreator::Create(provider_options))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(provider_options);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider() {
// The NNAPI EP uses a stub implementation on non-Android platforms so cannot be used to execute a model.
// Manually append an NNAPI EP instance to the session to unit test the GetCapability and Compile implementation.
#if defined(USE_NNAPI) && defined(__ANDROID__)
  return NnapiProviderFactoryCreator::Create(0, {})->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultVSINPUExecutionProvider() {
#if defined(USE_VSINPU)
  return VSINPUProviderFactoryCreator::Create()->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultRknpuExecutionProvider() {
#ifdef USE_RKNPU
  return RknpuProviderFactoryCreator::Create()->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_fast_math) {
#ifdef USE_ACL
  return ACLProviderFactoryCreator::Create(enable_fast_math)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_fast_math);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultArmNNExecutionProvider(bool enable_arena) {
#ifdef USE_ARMNN
  return ArmNNProviderFactoryCreator::Create(enable_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCoreMLExecutionProvider(bool use_mlprogram) {
  // To manually test CoreML model generation on a non-macOS platform, comment out the `&& defined(__APPLE__)` below.
  // The test will create a model but execution of it will obviously fail.
#if defined(USE_COREML) && defined(__APPLE__)
  // We want to run UT on CPU only to get output value without losing precision
  auto option = ProviderOptions();
  option[kCoremlProviderOption_MLComputeUnits] = "CPUOnly";

  if (use_mlprogram) {
    option[kCoremlProviderOption_ModelFormat] = "MLProgram";
  }

  return CoreMLProviderFactoryCreator::Create(option)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(use_mlprogram);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultSnpeExecutionProvider() {
#if defined(USE_SNPE)
  ProviderOptions provider_options_map;
  return SNPEProviderFactoryCreator::Create(provider_options_map)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultQnnExecutionProvider() {
#ifdef USE_QNN
  ProviderOptions provider_options_map;
  // Limit to CPU backend for now. TODO: Enable HTP emulator
  std::string backend_path = "./libQnnCpu.so";
#if defined(_WIN32) || defined(_WIN64)
  backend_path = "./QnnCpu.dll";
#endif
  provider_options_map["backend_path"] = backend_path;
  return QNNProviderFactoryCreator::Create(provider_options_map, nullptr)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> QnnExecutionProviderWithOptions(const ProviderOptions& options,
                                                                    const SessionOptions* session_options) {
#ifdef USE_QNN
  return QNNProviderFactoryCreator::Create(options, session_options)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(session_options);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultXnnpackExecutionProvider() {
#ifdef USE_XNNPACK
  return XnnpackProviderFactoryCreator::Create(ProviderOptions(), nullptr)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultWebGpuExecutionProvider(bool is_nhwc) {
#ifdef USE_WEBGPU
  ConfigOptions config_options{};
  // Disable storage buffer cache
  ORT_ENFORCE(config_options.AddConfigEntry(webgpu::options::kStorageBufferCacheMode,
                                            webgpu::options::kBufferCacheMode_Disabled)
                  .IsOK());
  if (!is_nhwc) {
    // Enable NCHW support
    ORT_ENFORCE(config_options.AddConfigEntry(webgpu::options::kPreferredLayout,
                                              webgpu::options::kPreferredLayout_NCHW)
                    .IsOK());
  }
  return WebGpuProviderFactoryCreator::Create(config_options)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(is_nhwc);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> WebGpuExecutionProviderWithOptions(const ConfigOptions& config_options) {
#ifdef USE_WEBGPU
  return WebGpuProviderFactoryCreator::Create(config_options)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(config_options);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCannExecutionProvider() {
#ifdef USE_CANN
  OrtCANNProviderOptions provider_options{};
  if (auto factory = CannProviderFactoryCreator::Create(&provider_options))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultDmlExecutionProvider() {
#ifdef USE_DML
  ConfigOptions config_options{};
  if (auto factory = DMLProviderFactoryCreator::CreateFromDeviceOptions(config_options, nullptr, false, false)) {
    return factory->CreateProvider();
  }
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultRocmExecutionProvider(bool) {
  return nullptr;
}
}  // namespace test
}  // namespace onnxruntime
