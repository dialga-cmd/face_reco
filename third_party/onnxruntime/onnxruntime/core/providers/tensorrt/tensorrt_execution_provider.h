// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <ctime>
#ifndef USE_CUDA_MINIMAL
#include <cudnn.h>
#else
typedef void* cudnnHandle_t;
typedef void* cublasHandle_t;
typedef void* cudnnStatus_t;
#endif
#include "core/providers/tensorrt/nv_includes.h"

#include <mutex>
#include "core/providers/cuda/cuda_graph.h"
#include "tensorrt_execution_provider_info.h"

namespace onnxruntime {

namespace tensorrt_env_vars {
static const std::string kMaxPartitionIterations = "ORT_TENSORRT_MAX_PARTITION_ITERATIONS";
static const std::string kMinSubgraphSize = "ORT_TENSORRT_MIN_SUBGRAPH_SIZE";
static const std::string kMaxWorkspaceSize = "ORT_TENSORRT_MAX_WORKSPACE_SIZE";
static const std::string kFP16Enable = "ORT_TENSORRT_FP16_ENABLE";
static const std::string kBF16Enable = "ORT_TENSORRT_BF16_ENABLE";
static const std::string kINT8Enable = "ORT_TENSORRT_INT8_ENABLE";
static const std::string kINT8CalibrationTableName = "ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME";
static const std::string kINT8UseNativeTensorrtCalibrationTable = "ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE";
static const std::string kDLAEnable = "ORT_TENSORRT_DLA_ENABLE";
static const std::string kDLACore = "ORT_TENSORRT_DLA_CORE";
static const std::string kDumpSubgraphs = "ORT_TENSORRT_DUMP_SUBGRAPHS";
static const std::string kEngineCacheEnable = "ORT_TENSORRT_ENGINE_CACHE_ENABLE";
static const std::string kCachePath = "ORT_TENSORRT_CACHE_PATH";
static const std::string kWeightStrippedEngineEnable = "ORT_TENSORRT_WEIGHT_STRIPPED_ENGINE_ENABLE";
static const std::string kOnnxModelFolderPath = "ORT_TENSORRT_ONNX_MODEL_FOLDER_PATH";
// As a timing cache can be used across multiple ONNX files it makes sense to have a separate cache path
static const std::string kTimingCachePath = "ORT_TENSORRT_GLOBAL_CACHE_PATH";
static const std::string kDecryptionEnable = "ORT_TENSORRT_ENGINE_DECRYPTION_ENABLE";
static const std::string kDecryptionLibPath = "ORT_TENSORRT_ENGINE_DECRYPTION_LIB_PATH";
static const std::string kForceSequentialEngineBuild = "ORT_TENSORRT_FORCE_SEQUENTIAL_ENGINE_BUILD";
static const std::string kContextMemorySharingEnable = "ORT_TENSORRT_CONTEXT_MEMORY_SHARING_ENABLE";
static const std::string kLayerNormFP32Fallback = "ORT_TENSORRT_LAYER_NORM_FP32_FALLBACK";
static const std::string kTimingCacheEnable = "ORT_TENSORRT_TIMING_CACHE_ENABLE";
static const std::string kForceTimingCache = "ORT_TENSORRT_FORCE_TIMING_CACHE_ENABLE";
static const std::string kDetailedBuildLog = "ORT_TENSORRT_DETAILED_BUILD_LOG_ENABLE";
static const std::string kBuildHeuristics = "ORT_TENSORRT_BUILD_HEURISTICS_ENABLE";
static const std::string kSparsityEnable = "ORT_TENSORRT_SPARSITY_ENABLE";
static const std::string kBuilderOptimizationLevel = "ORT_TENSORRT_BUILDER_OPTIMIZATION_LEVEL";
static const std::string kAuxiliaryStreams = "ORT_TENSORRT_AUXILIARY_STREAMS";
static const std::string kTacticSources = "ORT_TENSORRT_TACTIC_SOURCES";
static const std::string kExtraPluginLibPaths = "ORT_TENSORRT_EXTRA_PLUGIN_LIB_PATHS";
static const std::string kProfilesMinShapes = "ORT_TENSORRT_PROFILE_MIN_SHAPES";
static const std::string kProfilesMaxShapes = "ORT_TENSORRT_PROFILE_MAX_SHAPES";
static const std::string kProfilesOptShapes = "ORT_TENSORRT_PROFILE_OPT_SHAPES";
static const std::string kCudaGraphEnable = "ORT_TENSORRT_CUDA_GRAPH_ENABLE";
static const std::string kDumpEpContextModel = "ORT_DUMP_EP_CONTEXT_MODEL";
static const std::string kEpContextEmbedMode = "ORT_EP_CONTEXT_EMBED_MODE";
static const std::string kEpContextComputeCapabilityEnable = "ORT_EP_CONTEXT_COMPUTE_CAPABILITY_ENABLE";
static const std::string kEngineCachePrefix = "ORT_TENSORRT_CACHE_PREFIX";
static const std::string kOpTypesToExclude = "ORT_TENSORRT_OP_TYPES_TO_EXCLUDE";
// Old env variable for backward compatibility
static const std::string kEngineCachePath = "ORT_TENSORRT_ENGINE_CACHE_PATH";
}  // namespace tensorrt_env_vars

class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;

 public:
  TensorrtLogger(Severity verbosity = Severity::kWARNING)
      : verbosity_(verbosity) {}
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= verbosity_) {
      time_t rawtime = std::time(0);
      struct tm stm;
#ifdef _MSC_VER
      gmtime_s(&stm, &rawtime);
#else
      gmtime_r(&rawtime, &stm);
#endif
      char buf[256];
      strftime(&buf[0], 256,
               "%Y-%m-%d %H:%M:%S",
               &stm);
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR ? "  ERROR"
                                                                            : severity == Severity::kWARNING ? "WARNING"
                                                                            : severity == Severity::kINFO    ? "   INFO"
                                                                                                             : "UNKNOWN");
      if (severity <= Severity::kERROR) {
        LOGS_DEFAULT(ERROR) << "[" << buf << " " << sevstr << "] " << msg;
      } else {
        LOGS_DEFAULT(WARNING) << "[" << buf << " " << sevstr << "] " << msg;
      }
    }
  }
  void set_level(Severity verbosity) {
    verbosity_ = verbosity;
  }
  Severity get_level() const {
    return verbosity_;
  }
};

namespace tensorrt_ptr {

struct TensorrtInferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      delete obj;
    }
  }
};

template <typename T>
using unique_pointer = std::unique_ptr<T, TensorrtInferDeleter>;
};  // namespace tensorrt_ptr

//
// Class to allocate memory for outputs with data-dependent shapes. The sizes of those are unknown so pre-allocation is
// not possible.
//
class OutputAllocator : public nvinfer1::IOutputAllocator {
 public:
#if NV_TENSORRT_MAJOR >= 10
  void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment, cudaStream_t stream) noexcept override;
#else
  void* reallocateOutput(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept override;
#endif
  void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override;

  void* getBuffer() {
    return outputPtr;
  }

  std::vector<int64_t>& getOutputShape() {
    return output_shapes;
  }

  uint64_t getSize() {
    return allocated_size;
  }

  ~OutputAllocator() override {
    cudaFree(outputPtr);
  }

 private:
  void* outputPtr{nullptr};
  uint64_t allocated_size = 0;
  std::vector<int64_t> output_shapes;
};

/*
 * This map saves the dimension range of the shape of the shape tensor or execution tensor:
 * tensor name -> ( dimension -> [min, max, opt] )
 */
using ShapeRangesMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>>;

// Struct to hold user weights when ModelProtos are serialized with data.
class TensorrtUserWeights {
 public:
  TensorrtUserWeights(const std::string& name, const std::string& data) : name_(name), data_(data) {};

  const char* Name() const {
    return name_.c_str();
  };

  const void* Data() const {
    return static_cast<void const*>(data_.data());
  }

  int64_t Size() const {
    return static_cast<int64_t>(data_.size());
  }

 private:
  std::string name_{};
  std::string data_{};
};

// Information to construct kernel function state.
struct TensorrtFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  std::string fused_node_name;
  nvinfer1::IBuilder* builder;
  tensorrt_ptr::unique_pointer<nvonnxparser::IParser>* parser = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine>* engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext>* context = nullptr;
  std::unique_ptr<nvinfer1::INetworkDefinition>* network = nullptr;
  std::vector<std::unordered_map<std::string, size_t>> input_info;
  std::vector<std::unordered_map<std::string, size_t>> output_info;
  std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>> input_shape_ranges;
  std::mutex* tensorrt_mu_ptr = nullptr;
  bool fp16_enable = false;
  bool bf16_enable = false;
  bool int8_enable = false;
  bool int8_calibration_cache_available = false;
  bool dla_enable = false;
  int dla_core = 0;
  std::string trt_node_name_with_precision;
  bool engine_cache_enable = false;
  std::string engine_cache_path;
  nvinfer1::IRuntime* runtime = nullptr;
  std::vector<nvinfer1::IOptimizationProfile*> profiles;
  bool context_memory_sharing_enable = false;
  size_t* max_context_mem_size_ptr = nullptr;
  IAllocatorUniquePtr<void>* context_memory = nullptr;
  std::unordered_map<std::string, float> dynamic_range_map;
  bool engine_decryption_enable = false;
  int (*engine_decryption)(const char*, char*, size_t*) = nullptr;
  int (*engine_encryption)(const char*, char*, size_t) = nullptr;
  bool timing_cache_enable = true;
  std::string timing_cache_path;
  bool force_timing_cache = false;
  bool detailed_build_log = false;
  bool build_heuristics_enable = false;
  bool sparsity_enable = false;
  int builder_optimization_level = 3;
  int auxiliary_streams = -1;
  bool filter_tactic_sources = false;
  nvinfer1::TacticSources tactic_sources;
  bool cuda_graph_enable = 0;
  std::string cache_prefix;
  std::string cache_suffix;
  bool engine_hw_compatible = false;
  std::vector<nvinfer1::PreviewFeature> preview_features;
  std::unique_ptr<std::vector<TensorrtUserWeights>>* userWeights = nullptr;
};

// Minimum information to construct kernel function state for direct engine load code path
struct TensorrtShortFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  std::string fused_node_name;
  std::unique_ptr<nvinfer1::ICudaEngine>* engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext>* context = nullptr;
  std::vector<std::unordered_map<std::string, size_t>> input_info;
  std::vector<std::unordered_map<std::string, size_t>> output_info;
  bool context_memory_sharing_enable = false;
  size_t* max_context_mem_size_ptr = nullptr;
  IAllocatorUniquePtr<void>* context_memory = nullptr;
  std::mutex* tensorrt_mu_ptr = nullptr;
};

// Holds important information for building valid ORT graph.
struct SubGraphContext {
  std::unordered_set<std::string> output_args;
  std::unordered_map<std::string, const NodeArg*> inputs_and_initializers;
  std::unordered_map<std::string, const NodeArg*> manually_added_graph_inputs;
};

using SubGraphContextMap = std::unordered_map<std::string, std::unique_ptr<SubGraphContext>>;
using DDSOutputAllocatorMap = std::unordered_map<std::string, std::unique_ptr<OutputAllocator>>;
std::string GetWeightRefittedEnginePath(std::string engine_cache_path);

// Logical device representation.
class TensorrtExecutionProvider : public IExecutionProvider {
 public:
  explicit TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info);
  virtual ~TensorrtExecutionProvider();

  cublasHandle_t PerThreadDefaultCublasHandle() {
    return GetPerThreadContext().CublasHandle();
  }

  cudnnHandle_t PerThreadDefaultCudnnHandle() {
    return GetPerThreadContext().CudnnHandle();
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph,
                const IKernelLookup& /*kernel_lookup*/,
                const GraphOptimizerRegistry& graph_optimizer_registry,
                IResourceAccountant* /* resource_accountant */) const override;

  int GetDeviceId() const { return device_id_; }

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;
  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

  ProviderOptions GetProviderOptions() const override {
    return TensorrtExecutionProviderInfo::ToProviderOptions(info_);
  }

  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;

  void GetCustomOpDomainList(std::vector<OrtCustomOpDomain*>& custom_op_domain_list) const override;

  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured(int graph_annotation_id) const override;
  Status ReplayGraph(int graph_annotation_id) override;

  static common::Status RefitEngine(std::string onnx_model_filename,
                                    std::string& onnx_model_folder_path,
                                    std::string& weight_stripped_engine_cath_path,
                                    bool path_check,
                                    const void* onnx_model_bytestream,
                                    size_t onnx_model_bytestream_size,
                                    const void* onnx_external_data_bytestream,
                                    size_t onnx_external_data_bytestream_size,
                                    nvinfer1::ICudaEngine* trt_engine,
                                    bool serialize_refitted_engine,
                                    bool detailed_build_log);

 private:
  mutable TensorrtExecutionProviderInfo info_;
  bool external_stream_ = false;
  cudaStream_t stream_ = nullptr;
  int max_partition_iterations_ = 1000;
  size_t min_subgraph_size_ = 1;
  size_t max_workspace_size_ = 0;
  bool fp16_enable_ = false;
  bool bf16_enable_ = false;
  bool int8_enable_ = false;
  bool dla_enable_ = false;
  int dla_core_ = 0;
  bool force_sequential_engine_build_ = false;
  std::string int8_calibration_cache_name_;
  bool int8_calibration_cache_available_ = false;
  bool int8_use_native_tensorrt_calibration_table_ = false;
  bool dump_subgraphs_ = false;
  bool engine_cache_enable_ = false;
  bool weight_stripped_engine_enable_ = false;
  bool weight_stripped_engine_refit_ = false;
  std::string onnx_model_folder_path_;
  const void* onnx_model_bytestream_;
  size_t onnx_model_bytestream_size_;
  const void* onnx_external_data_bytestream_ = nullptr;
  size_t onnx_external_data_bytestream_size_ = 0;
  bool build_heuristics_enable_ = false;
  bool sparsity_enable_ = false;
  int builder_optimization_level_ = 3;
  int auxiliary_streams_ = -1;
  std::string tactic_sources_;
  std::string global_cache_path_, cache_path_, engine_decryption_lib_path_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
  std::mutex tensorrt_mu_;
  int device_id_;
  std::string compute_capability_;
  bool context_memory_sharing_enable_ = false;
  bool layer_norm_fp32_fallback_ = false;
  size_t max_ctx_mem_size_ = 0;
  IAllocatorUniquePtr<void> context_memory_ = nullptr;
  mutable char model_path_[4096] = {};  // Reserved for max path length
  bool engine_decryption_enable_ = false;
  int (*engine_decryption_)(const char*, char*, size_t*) = nullptr;
  int (*engine_encryption_)(const char*, char*, size_t) = nullptr;
  bool timing_cache_enable_ = false;
  bool force_timing_cache_match_ = false;
  bool detailed_build_log_ = false;
  bool cuda_graph_enable_ = false;
  std::string cache_prefix_;
  bool engine_hw_compatible_ = false;
  std::string op_types_to_exclude_;
  std::vector<nvinfer1::PreviewFeature> preview_features_;
  bool load_user_initializer_ = false;

  // The format is as for TENSORRT_VERSION: (MAJOR * 100 + MINOR) * 100 + PATCH
  int32_t trt_version_;
  int32_t cuda_version_;

  // The OrtAllocator object will be get during ep compute time
  // and should be kept for the lifetime of TRT EP object.
  OrtAllocator* alloc_ = nullptr;

  // For create/dump EP context node model
  bool dump_ep_context_model_ = false;
  std::string ep_context_file_path_;
  int ep_context_embed_mode_ = 0;
  std::string ctx_model_path_;
  std::string ep_cache_context_attr_;
  std::string engine_cache_relative_path_to_context_model_dir;
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto_ = ONNX_NAMESPACE::ModelProto::Create();

  std::unordered_set<std::string> control_flow_op_set_ = {"If", "Loop", "Scan"};
  mutable std::unordered_map<std::string, std::unique_ptr<SubGraphContext>> subgraph_context_map_;

  mutable std::unique_ptr<nvinfer1::IBuilder> builder_;

  // Following maps that hold TRT objects will be accessible by different threads if ORT is using multithreading.
  // In general, TensorRT objects are not thread safe; accesses to an object from different threads must be serialized by the client.
  // But there are still some thread safe operations, please see here https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
  // For those non thread safe operations, TRT EP uses (1) lock_guard or (2) PerThreadContext to make sure synchronization.
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvonnxparser::IParser>> parsers_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::ICudaEngine>> engines_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::IExecutionContext>> contexts_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::IBuilder>> builders_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::INetworkDefinition>> networks_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> input_info_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> output_info_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_min_shapes_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_max_shapes_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_opt_shapes_;
  std::unordered_map<std::string, ShapeRangesMap> input_shape_ranges_;  // The profile shape ranges that the engine is built with
  std::unordered_map<std::string, std::vector<nvinfer1::IOptimizationProfile*>> profiles_;
  std::unordered_map<std::string, DDSOutputAllocatorMap> dds_output_allocator_maps_;
  std::unordered_map<std::string, std::unique_ptr<std::vector<TensorrtUserWeights>>> weights_;  // User provided weights.

  // for external stream, we need to create its cudnn/cublass handle before cuda EP enable cuda graph capture
  cudnnHandle_t external_cudnn_handle_ = nullptr;
  cublasHandle_t external_cublas_handle_ = nullptr;

  // Call cudaStreamSynchronize() after TRT enqueueV3()
  mutable bool sync_stream_after_enqueue_ = true;

  CUDAGraph cuda_graph_;
  bool is_graph_captured_ = false;
  int regular_run_count_before_graph_capture_ = 0;
  // There is chance (currently only happens in CUDA EP) that the second regular run allocates GPU memory for causes like:
  // (1) memory pattern is enabled. (2) arena allocation for stream.
  // Since no GPU memory allocation is allowed during graph capturing, we need at least two regular runs
  // to allocate enough memory in Arena before graph capturing.
  const int min_num_runs_before_cuda_graph_capture_ = 1;  // required min regular runs before graph capture for the necessary memory allocations.

  // [Note] We don't use PerThreadContext for now since it has issue with multithreading
  //
  // TRT or CUDA objects that must be maintained on a per thread basis will be put under this PerThreadContext data structure.
  // For example, TensorRT execution context and CUDA graph are the ones to be put here.
  class PerThreadContext final {
   public:
    PerThreadContext(OrtDevice::DeviceId device_id, bool has_user_compute_stream, cudaStream_t stream);
    ~PerThreadContext();

    cublasHandle_t CublasHandle() const {
      return external_cublas_handle_;
    }

    cudnnHandle_t CudnnHandle() const {
      return external_cudnn_handle_;
    }

    bool IsTensorRTContextInMap(std::string fused_node);
    nvinfer1::IExecutionContext& GetTensorRTContext(std::string fused_node);
    bool UpdateTensorRTContext(std::string fused_node, std::unique_ptr<nvinfer1::IExecutionContext> context);
    void ResetTensorRTContext(std::string fused_node);
    bool CompareProfileShapes(std::string fused_node, ShapeRangesMap& shape_ranges);
    void UpdateProfileShapes(std::string fused_node, ShapeRangesMap& shape_ranges);

    void InitCUDAGraph();
    void SetGraphStream(cudaStream_t stream);
    bool IsGraphCaptureAllowed() const;
    void CaptureBegin(int graph_annotation_id);
    void CaptureEnd(int graph_annotation_id);
    bool IsGraphCaptured(int graph_annotation_id) const;
    Status ReplayGraph(int graph_annotation_id);
    void IncrementRegularRunCountBeforeGraphCapture();

   private:
    cudnnHandle_t external_cudnn_handle_ = nullptr;
    cublasHandle_t external_cublas_handle_ = nullptr;

    // Maintaining execution context on a per thread basis is suggested by TRT doc.
    // Also, for enqueueV2() in execution context, to perform inference concurrently in multiple streams, use one execution context per stream.
    // ORT multi-streams feature uses one stream for one thread, therefore maintaining execution context on a per thread basis is necessary for TRT EP,
    // otherwise it may result in undefined behavior or synchronization issues.
    //
    // See more details here:
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
    // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#a63cd95430852038ce864e17c670e0b36
    std::unordered_map<std::string, std::unique_ptr<nvinfer1::IExecutionContext>> trt_context_map_;

    // The profile shape ranges for the engine that the execution context maintained by the PerThreadContext is built with.
    // TRT EP needs this info to determine whether to rebuild the execution context.
    std::unordered_map<std::string, ShapeRangesMap> input_shape_ranges_;

    // Cuda graph with multi threads will be supported in the future, so cuda_graph_ is put under PerThreadContext.
    // ORT TRT only supports CUDA graph when whole model is supported by TRT, so simply maintaining a CUDAGraph instance is enough (no need to maintain one CUDAGraph instance per TRT subgraph)
    CUDAGraph cuda_graph_;
    bool is_graph_captured_ = false;
    int regular_run_count_before_graph_capture_ = 0;
    // There is chance (currently only happens in CUDA EP) that the second regular run allocates GPU memory for causes like:
    // (1) memory pattern is enabled. (2) arena allocation for stream.
    // Since no GPU memory allocation is allowed during graph capturing, we need at least two regular runs
    // to allocate enough memory in Arena before graph capturing.
    const int min_num_runs_before_cuda_graph_capture_ = 1;  // required min regular runs before graph capture for the necessary memory allocations.
  };

  using PerThreadContextMap = std::unordered_map<const TensorrtExecutionProvider*, std::weak_ptr<PerThreadContext>>;
  // thread local PerThreadContext cache

  struct ContextCacheHolder {
    ContextCacheHolder() {
      // Keep a weak pointer to the object, if the weak pointer can be locked, then the shared pointer is still around, so we can reset it
      RunOnUnload([&, weak_p_ = std::weak_ptr<PerThreadContextMap>(p)] {
        if (auto lock = weak_p_.lock()) {
          p.reset();
        }
      });
    }

    std::shared_ptr<PerThreadContextMap> p = std::make_shared<PerThreadContextMap>();
  };

  static const std::shared_ptr<PerThreadContextMap>& PerThreadContextCache() {
    thread_local const ContextCacheHolder per_thread_context_cache;
    return per_thread_context_cache.p;
  }

  struct PerThreadContextState {
    // contexts that are currently active
    std::set<std::shared_ptr<PerThreadContext>, std::owner_less<std::shared_ptr<PerThreadContext>>> active_contexts;
    // contexts available for reuse
    std::vector<std::shared_ptr<PerThreadContext>> retired_context_pool;
    // weak references to thread local caches from which this TensorrtExecutionProvider instance's entry should be removed
    // upon destruction
    std::set<std::weak_ptr<PerThreadContextMap>, std::owner_less<std::weak_ptr<PerThreadContextMap>>>
        caches_to_update_on_destruction;
    // synchronizes access to PerThreadContextState members
    std::mutex mutex;
  };

  // The execution provider maintains the PerThreadContexts in this structure.
  // Synchronization is required to update the contained structures.
  // On the other hand, access to an individual PerThreadContext is assumed to be from a single thread at a time,
  // so synchronization is not required for that.
  mutable PerThreadContextState context_state_;

  PerThreadContext& GetPerThreadContext() const;
  void ReleasePerThreadContext() const;

  /**Get IndexedSubGraph based on node list of the subgraph*/
  std::unique_ptr<IndexedSubGraph> GetSubGraph(SubGraph_t graph_nodes_index,
                                               const GraphViewer& graph, const HashValue& model_hash, int subgraph_index) const;

  /**
  Get TensorRT supported node lists by calling Onnx-TensorRT parser recursively. Since each time the parser
  can only detect first unsupported node failure, it needs to wait for Onnxruntime to partition the graph
  and then detect next failure again. If there are too many iterations, which means many nodes in the graph
  are not supported by TensorRT, the process will be terminated and the whole graph is simply assigned to
  other execution provider.
  */
  SubGraphCollection_t GetSupportedList(SubGraphCollection_t supported_nodes_list, int iterations, const int max_iterations,
                                        const GraphViewer& graph, bool* early_termination) const;

  bool DetectTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const GraphViewer& graph, const HashValue& model_hash, bool remove_cycles = true) const;

  /**
  Get a unique_lock object to control the concurrency behavior.
  Every api call not in the thread-safe operations(https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading)
  should be protected by a lock when invoked by multiple threads concurrently.
  */
  std::unique_lock<std::mutex> GetApiLock() const;

  /**Check the graph is the subgraph of control flow op*/
  bool IsSubGraphOfControlFlowOp(const GraphViewer& graph) const;

  /**Check whether all the nodes of the graph are assigned to specific ep*/
  bool AllNodesAssignedToSpecificEP(const GraphViewer& graph, const std::string& provider_type) const;

  /**Check whether all the nodes of subgraph are supported*/
  bool IsSubGraphFullySupported(SubGraphCollection_t supported_nodes_vector, const int number_of_ort_nodes) const;

  /**
   * Set inputs, initializers and outputs for all subgraphs during TensorrtExecutionProvider::GetSupportedList()
   * and save those information in subgraph context data structure. It's useful for building a valid graph and
   * make Graph::Resolve() happy especially when dealing with nested control-flow op graph.
   */
  void BuildSubGraphContext(Graph& build_graph) const;

  /**
   * Set outer scope values for subgraphs and add thoes values as top-level graph's inputs if needed.
   */
  void SetGraphOuterScopeValuesAndInputs(Graph& build_graph, const Graph& graph) const;

  /**
   * If ORT TRT manually sets graph input in TensorrtExecutionProvider::SetGraphOuterScopeValuesAndInputs(),
   * we have to manully set all the graph inputs in order to pass Graph::Resolve().
   */
  void SetAllGraphInputs(Graph& graph) const;

  /**
   * The newly-built graph has not yet being resolved by Graph::Resolve(), so we can't leverage
   * Graph::ResolveContext::IsInputInitializerOrOutput(). We have to implement this fuction again.
   */
  bool IsInputInitializerOrOutput(const Graph& graph, const std::string& name, bool check_ancestors) const;

  /**
   * The newly-built graph has not yet being resolved by Graph::Resolve(), so we can't leverage
   * Graph::ResolveContext::IsOuterScopeValue(). We have to implement this fuction again.
   */
  bool IsOuterScopeValue(const Graph& graph, const std::string& name) const;

  /**
   * The newly-built graph has not yet being resolved by Graph::Resolve(), so we can't leverage
   * Graph::ResolveContext::IsLocalValue(). We have to implement this fuction again.
   */
  bool IsLocalValue(const Graph& graph, const std::string& name) const;

  /**
   * Create a vector of NodeComputeInfo instances directly from "TRT engine" wrapped onnx model without
   * going through the time-consuming processes of model parsing and engine building.
   */
  Status CreateNodeComputeInfoFromPrecompiledEngine(const GraphViewer& graph_body_viewer,
                                                    const Node& fused_node,
                                                    std::unordered_map<std::string, size_t>& input_map,
                                                    std::unordered_map<std::string, size_t>& output_map,
                                                    std::vector<NodeComputeInfo>& node_compute_funcs);

  /**
   * Create a vector of NodeComputeInfo instances from graph.
   */
  Status CreateNodeComputeInfoFromGraph(const GraphViewer& graph_body_viewer,
                                        const Node& fused_node,
                                        std::unordered_map<std::string, size_t>& input_map,
                                        std::unordered_map<std::string, size_t>& output_map,
                                        std::vector<NodeComputeInfo>& node_compute_funcs);

  bool IsGraphCaptureAllowed() const;
  void CaptureBegin(int graph_annotation_id);
  void CaptureEnd(int graph_annotation_id);
  void IncrementRegularRunCountBeforeGraphCapture();

  /**
   * Get the pointer to the IBuilder instance.
   * This function only creates the instance at the first time it's being called."
   */
  nvinfer1::IBuilder* GetBuilder(TensorrtLogger& trt_logger) const;

  /**
   *  This is the helper function for ConstantFoldingDQ graph transformer.
   *
   *  It selects the qualified/required DQ node to be optimized as well as provides a mapping table
   *  to help TRT EP later include the DQ node which is filtered out by TRT parser.
   */
  void SelectQualifiedDQNode(const GraphViewer& graph,
                             std::unordered_set<NodeIndex>& selection_node_set,
                             std::unordered_map<NodeIndex, NodeIndex>& consumer_to_dq) const;

  /**
   * This function returns an optimization ComputeCapability that is limited to:
   *  1. the DQ nodes in this individual TRT ComputeCapability
   *  2. the DQ nodes that are qualified and selected by TRT EP
   *
   * It also needs to make sure the DQ nodes is a subset of the complete list of DQ nodes to optimize in original selection ComputeCapability.
   * Finally, copy the optimization function from the original selection ComputeCapability.
   */
  std::unique_ptr<ComputeCapability> CreateOptimizationComputeCapability(ComputeCapability* selection_cc,
                                                                         std::unordered_set<NodeIndex>& trt_selection_node_set,
                                                                         ComputeCapability* trt_cc) const;
  /**
   * This function helps add back the DQ nodes that are filtered out by TRT parser.
   * The reason is the DQ nodes can be optimized and dequantized by applying ConstantFoldingDQ optimizer by ORT L2+ optimization.
   */
  void UpdateSupportedNodeVectorForDQ(const GraphViewer& graph,
                                      SubGraph_t& supported_node_vector,
                                      SubGraphCollection_t& supported_nodes_vector,
                                      std::unordered_map<NodeIndex, NodeIndex> consumer_to_dq) const;
};
}  // namespace onnxruntime
