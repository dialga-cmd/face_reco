/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#include "core/platform/env.h"

#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <ftw.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#if !defined(_AIX)
#include <sys/syscall.h>
#endif
#include <unistd.h>

#include <iostream>
#include <optional>
#include <thread>
#include <utility>  // for std::forward
#include <vector>

// We can not use CPUINFO if it is not supported and we do not want to used
// it on certain platforms because of the binary size increase.
// We could use it to find out the number of physical cores for certain supported platforms
#if defined(CPUINFO_SUPPORTED) && !defined(__APPLE__) && !defined(__ANDROID__) && !defined(__wasm__) && !defined(_AIX)
#include <cpuinfo.h>
#define ORT_USE_CPUINFO
#endif

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)
#include <sys/sysctl.h>
#endif

#include "core/common/common.h"
#include <gsl/gsl>
#include "core/common/logging/logging.h"
#include "core/common/narrow.h"
#include "core/platform/scoped_resource.h"
#include "core/platform/EigenNonBlockingThreadPool.h"

namespace onnxruntime {

namespace {

constexpr int OneMillion = 1000000;

static void UnmapFile(void* addr, size_t len) noexcept {
  int ret = munmap(addr, len);
  if (ret != 0) {
    auto [err_no, err_msg] = GetErrnoInfo();
    LOGS_DEFAULT(ERROR) << "munmap failed. error code: " << err_no << " error msg: " << err_msg;
  }
}

struct FileDescriptorTraits {
  using Handle = int;
  static Handle GetInvalidHandleValue() { return -1; }
  static void CleanUp(Handle h) {
    if (close(h) == -1) {
      auto [err_no, err_msg] = GetErrnoInfo();
      LOGS_DEFAULT(ERROR) << "Failed to close file descriptor " << h << " - error code: " << err_no
                          << " error msg: " << err_msg;
    }
  }
};

// Note: File descriptor cleanup may fail but this class doesn't expose a way to check if it failed.
//       If that's important, consider using another cleanup method.
using ScopedFileDescriptor = ScopedResource<FileDescriptorTraits>;

// non-macro equivalent of TEMP_FAILURE_RETRY, described here:
// https://www.gnu.org/software/libc/manual/html_node/Interrupted-Primitives.html
template <typename TFunc, typename... TFuncArgs>
long int TempFailureRetry(TFunc retriable_operation, TFuncArgs&&... args) {
  long int result;
  do {
    result = retriable_operation(std::forward<TFuncArgs>(args)...);
  } while (result == -1 && errno == EINTR);
  return result;
}

// nftw() callback to remove a file
int nftw_remove(
    const char* fpath, const struct stat* /*sb*/,
    int /*typeflag*/, struct FTW* /*ftwbuf*/) {
  const auto result = remove(fpath);
  if (result != 0) {
    auto [err_no, err_msg] = GetErrnoInfo();
    LOGS_DEFAULT(WARNING) << "remove() failed. Error code: " << err_no << " error msg: " << err_msg
                          << ", path: " << fpath;
  }
  return result;
}

template <typename T>
struct Freer {
  void operator()(T* p) { ::free(p); }
};

using MallocdStringPtr = std::unique_ptr<char, Freer<char> >;

class PosixThread : public EnvThread {
 private:
  struct Param {
    const ORTCHAR_T* name_prefix;
    int index;
    unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param);
    Eigen::ThreadPoolInterface* param;
    std::optional<LogicalProcessors> affinity;

    Param(const ORTCHAR_T* name_prefix1,
          int index1,
          unsigned (*start_address1)(int id, Eigen::ThreadPoolInterface* param),
          Eigen::ThreadPoolInterface* param1)
        : name_prefix(name_prefix1),
          index(index1),
          start_address(start_address1),
          param(param1) {}
  };

 public:
  PosixThread(const ORTCHAR_T* name_prefix, int index,
              unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param), Eigen::ThreadPoolInterface* param,
              const ThreadOptions& thread_options) {
    ORT_ENFORCE(index >= 0, "Negative thread index is not allowed");
    custom_create_thread_fn = thread_options.custom_create_thread_fn;
    custom_thread_creation_options = thread_options.custom_thread_creation_options;
    custom_join_thread_fn = thread_options.custom_join_thread_fn;

    auto param_ptr = std::make_unique<Param>(name_prefix, index, start_address, param);
    if (narrow<size_t>(index) < thread_options.affinities.size()) {
      param_ptr->affinity = thread_options.affinities[index];
    }

    if (custom_create_thread_fn) {
      custom_thread_handle = custom_create_thread_fn(custom_thread_creation_options, CustomThreadMain, param_ptr.get());
      if (!custom_thread_handle) {
        ORT_THROW("custom_create_thread_fn returned invalid handle.");
      }
      param_ptr.release();
    } else {
      pthread_attr_t attr;
      int s = pthread_attr_init(&attr);
      if (s != 0) {
        auto [err_no, err_msg] = GetErrnoInfo();
        ORT_THROW("pthread_attr_init failed, error code: ", err_no, " error msg: ", err_msg);
      }

      size_t stack_size = thread_options.stack_size;
      if (stack_size > 0) {
        s = pthread_attr_setstacksize(&attr, stack_size);
        if (s != 0) {
          auto [err_no, err_msg] = GetErrnoInfo();
          ORT_THROW("pthread_attr_setstacksize failed, error code: ", err_no, " error msg: ", err_msg);
        }
      }

      s = pthread_create(&hThread, &attr, ThreadMain, param_ptr.get());
      if (s != 0) {
        auto [err_no, err_msg] = GetErrnoInfo();
        ORT_THROW("pthread_create failed, error code: ", err_no, " error msg: ", err_msg);
      }
      param_ptr.release();
      // Do not throw beyond this point so we do not lose thread handle and then not being able to join it.
    }
  }

  ~PosixThread() override {
    if (custom_thread_handle) {
      custom_join_thread_fn(custom_thread_handle);
      custom_thread_handle = nullptr;
    } else {
      void* res;
#ifdef NDEBUG
      pthread_join(hThread, &res);
#else
      int ret = pthread_join(hThread, &res);
      assert(ret == 0);
#endif
    }
  }

 private:
  static void* ThreadMain(void* param) {
    std::unique_ptr<Param> p(static_cast<Param*>(param));
    ORT_TRY {
#if !defined(__APPLE__) && !defined(__ANDROID__) && !defined(__wasm__) && !defined(_AIX)
      if (p->affinity.has_value() && !p->affinity->empty()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (auto id : *p->affinity) {
          if (id > -1 && id < CPU_SETSIZE) {
            CPU_SET(id, &cpuset);
          } else {
            // Logical processor id starts from 0 internally, but in ort API, it starts from 1,
            // that's why id need to increase by 1 when logging.
            LOGS_DEFAULT(ERROR) << "cpu " << id + 1 << " does not exist, skipping it for affinity setting";
          }
        }
        auto ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (0 == ret) {
          LOGS_DEFAULT(VERBOSE) << "pthread_setaffinity_np succeed for thread: " << syscall(SYS_gettid)
                                << ", index: " << p->index
                                << ", mask: " << *p->affinity;
        } else {
          errno = ret;
          auto [err_no, err_msg] = GetErrnoInfo();
          LOGS_DEFAULT(ERROR) << "pthread_setaffinity_np failed for thread: " << syscall(SYS_gettid)
                              << ", index: " << p->index
                              << ", mask: " << *p->affinity
                              << ", error code: " << err_no << " error msg: " << err_msg
                              << ". Specify the number of threads explicitly so the affinity is not set.";
        }
      }
#endif
      // Ignore the returned value for now
      p->start_address(p->index, p->param);
    }
    ORT_CATCH(...) {
      // Ignore exceptions
    }
    return nullptr;
  }
  static void CustomThreadMain(void* param) {
    ThreadMain(param);
  }
  pthread_t hThread;
};

class PosixEnv : public Env {
 public:
  static PosixEnv& Instance() {
    static PosixEnv default_env;
    return default_env;
  }

  EnvThread* CreateThread(const ORTCHAR_T* name_prefix, int index,
                          unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                          Eigen::ThreadPoolInterface* param, const ThreadOptions& thread_options) override {
    return new PosixThread(name_prefix, index, start_address, param, thread_options);
  }

  // we are guessing the number of phys cores based on a popular HT case (2 logical proc per core)
  static int DefaultNumCores() {
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
  }

  // Return the number of physical cores
  int GetNumPhysicalCpuCores() const override {
#ifdef ORT_USE_CPUINFO
    if (cpuinfo_available_) {
      return narrow<int>(cpuinfo_get_cores_count());
    }
#endif  // ORT_USE_CPUINFO
    return DefaultNumCores();
  }

  std::vector<LogicalProcessors> GetDefaultThreadAffinities() const override {
    std::vector<LogicalProcessors> ret;
#ifdef ORT_USE_CPUINFO
    if (cpuinfo_available_) {
      auto num_phys_cores = cpuinfo_get_cores_count();
      ret.reserve(num_phys_cores);
      for (uint32_t i = 0; i < num_phys_cores; ++i) {
        const auto* core = cpuinfo_get_core(i);
        LogicalProcessors th_aff;
        th_aff.reserve(core->processor_count);
        auto log_proc_idx = core->processor_start;
        for (uint32_t count = 0; count < core->processor_count; count++, ++log_proc_idx) {
          const auto* log_proc = cpuinfo_get_processor(log_proc_idx);
          th_aff.push_back(log_proc->linux_id);
        }
        ret.push_back(std::move(th_aff));
      }
    }
#endif
    // Just the size of the thread-pool
    if (ret.empty()) {
      ret.resize(GetNumPhysicalCpuCores());
    }
    return ret;
  }

  int GetL2CacheSize() const override {
#ifdef _SC_LEVEL2_CACHE_SIZE
    return static_cast<int>(sysconf(_SC_LEVEL2_CACHE_SIZE));
#else
    int value = 0;  // unknown
#if (defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)) && defined(HW_L2CACHESIZE)
    int mib[2] = {CTL_HW, HW_L2CACHESIZE};
    size_t len = sizeof(value);
    if (sysctl(mib, 2, &value, &len, NULL, 0) < 0) {
      return -1;  // error
    }
#endif
    return value;
#endif
  }

  void SleepForMicroseconds(int64_t micros) const override {
    while (micros > 0) {
      timespec sleep_time;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 0;

      if (micros >= OneMillion) {
        sleep_time.tv_sec = static_cast<time_t>(std::min<int64_t>(micros / OneMillion,
                                                                  std::numeric_limits<time_t>::max()));
        micros -= static_cast<int64_t>(sleep_time.tv_sec) * OneMillion;
      }
      if (micros < OneMillion) {
        sleep_time.tv_nsec = static_cast<decltype(timespec::tv_nsec)>(1000 * micros);
        micros = 0;
      }
      while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
        // Ignore signals and wait for the full interval to elapse.
      }
    }
  }

  PIDType GetSelfPid() const override {
    return getpid();
  }

  Status GetFileLength(const PathChar* file_path, size_t& length) const override {
    ScopedFileDescriptor file_descriptor{open(file_path, O_RDONLY)};
    return GetFileLength(file_descriptor.Get(), length);
  }

  common::Status GetFileLength(int fd, /*out*/ size_t& file_size) const override {
    using namespace common;
    if (fd < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid fd was supplied: ", fd);
    }

    struct stat buf;
    int rc = fstat(fd, &buf);
    if (rc < 0) {
      return ReportSystemError("fstat", "");
    }

    if (buf.st_size < 0) {
      return ORT_MAKE_STATUS(SYSTEM, FAIL, "Received negative size from stat call");
    }

    if (static_cast<unsigned long long>(buf.st_size) > std::numeric_limits<size_t>::max()) {
      return ORT_MAKE_STATUS(SYSTEM, FAIL, "File is too large.");
    }

    file_size = static_cast<size_t>(buf.st_size);
    return Status::OK();
  }

  Status ReadFileIntoBuffer(const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
                            gsl::span<char> buffer) const override {
    ORT_RETURN_IF_NOT(file_path, "file_path == nullptr");
    ORT_RETURN_IF_NOT(offset >= 0, "offset < 0");
    ORT_RETURN_IF_NOT(length <= buffer.size(), "length > buffer.size()");

    ScopedFileDescriptor file_descriptor{open(file_path, O_RDONLY)};
    if (!file_descriptor.IsValid()) {
      return ReportSystemError("open", file_path);
    }

    if (length == 0)
      return Status::OK();

    if (offset > 0) {
      const FileOffsetType seek_result = lseek(file_descriptor.Get(), offset, SEEK_SET);
      if (seek_result == -1) {
        return ReportSystemError("lseek", file_path);
      }
    }

    size_t total_bytes_read = 0;
    while (total_bytes_read < length) {
      constexpr size_t k_max_bytes_to_read = 1 << 30;  // read at most 1GB each time
      const size_t bytes_remaining = length - total_bytes_read;
      const size_t bytes_to_read = std::min(bytes_remaining, k_max_bytes_to_read);

      const ssize_t bytes_read =
          TempFailureRetry(read, file_descriptor.Get(), buffer.data() + total_bytes_read, bytes_to_read);

      if (bytes_read == -1) {
        return ReportSystemError("read", file_path);
      }

      if (bytes_read == 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ReadFileIntoBuffer - unexpected end of file. ", "File: ", file_path,
                               ", offset: ", offset, ", length: ", length);
      }

      total_bytes_read += bytes_read;
    }

    return Status::OK();
  }

  Status MapFileIntoMemory(const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
                           MappedMemoryPtr& mapped_memory) const override {
    ORT_RETURN_IF_NOT(file_path, "file_path == nullptr");
    ORT_RETURN_IF_NOT(offset >= 0, "offset < 0");

    ScopedFileDescriptor file_descriptor{open(file_path, O_RDONLY)};
    if (!file_descriptor.IsValid()) {
      return ReportSystemError("open", file_path);
    }

    if (length == 0) {
      mapped_memory = MappedMemoryPtr{};
      return Status::OK();
    }

    static const size_t page_size = narrow<size_t>(sysconf(_SC_PAGESIZE));
    const FileOffsetType offset_to_page = offset % static_cast<FileOffsetType>(page_size);
    const size_t mapped_length = length + static_cast<size_t>(offset_to_page);
    const FileOffsetType mapped_offset = offset - offset_to_page;
    void* const mapped_base =
        mmap(nullptr, mapped_length, PROT_READ | PROT_WRITE, MAP_PRIVATE, file_descriptor.Get(), mapped_offset);

    if (mapped_base == MAP_FAILED) {
      return ReportSystemError("mmap", file_path);
    }

    mapped_memory =
        MappedMemoryPtr{reinterpret_cast<char*>(mapped_base) + offset_to_page,
                        [mapped_base, mapped_length](void*) {
                          UnmapFile(mapped_base, mapped_length);
                        }};

    return Status::OK();
  }

  static common::Status ReportSystemError(const char* operation_name, const std::string& path) {
    auto [err_no, err_msg] = GetErrnoInfo();
    std::ostringstream oss;
    oss << operation_name << " file \"" << path << "\" failed: " << err_msg;
    return common::Status(common::SYSTEM, err_no, oss.str());
  }

  bool FolderExists(const std::string& path) const override {
    struct stat sb;
    if (stat(path.c_str(), &sb)) {
      return false;
    }
    return S_ISDIR(sb.st_mode);
  }

  bool FileExists(const std::string& path) const override {
    struct stat sb;
    if (stat(path.c_str(), &sb)) {
      return false;
    }
    return S_ISREG(sb.st_mode);
  }

  common::Status CreateFolder(const std::string& path) const override {
    size_t pos = 0;
    do {
      pos = path.find_first_of("\\/", pos + 1);
      std::string directory = path.substr(0, pos);
      if (FolderExists(directory.c_str())) {
        continue;
      }
      if (mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
        return common::Status(common::SYSTEM, errno);
      }
    } while (pos != std::string::npos);
    return Status::OK();
  }

  common::Status DeleteFolder(const PathString& path) const override {
    const auto result = nftw(
        path.c_str(), &nftw_remove, 32, FTW_DEPTH | FTW_PHYS);
    ORT_RETURN_IF_NOT(result == 0, "DeleteFolder(): nftw() failed with error: ", result);
    return Status::OK();
  }

  common::Status FileOpenRd(const std::string& path, /*out*/ int& fd) const override {
    fd = open(path.c_str(), O_RDONLY);
    if (0 > fd) {
      return ReportSystemError("open", path);
    }
    return Status::OK();
  }

  common::Status FileOpenWr(const std::string& path, /*out*/ int& fd) const override {
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (0 > fd) {
      return ReportSystemError("open", path);
    }
    return Status::OK();
  }

  common::Status FileClose(int fd) const override {
    int ret = close(fd);
    if (0 != ret) {
      return ReportSystemError("close", "");
    }
    return Status::OK();
  }

  common::Status GetCanonicalPath(
      const PathString& path,
      PathString& canonical_path) const override {
    MallocdStringPtr canonical_path_cstr{realpath(path.c_str(), nullptr), Freer<char>()};
    if (!canonical_path_cstr) {
      return ReportSystemError("realpath", path);
    }
    canonical_path.assign(canonical_path_cstr.get());
    return Status::OK();
  }

  common::Status LoadDynamicLibrary(const PathString& library_filename, bool global_symbols, void** handle) const override {
    dlerror();  // clear any old error_str
    *handle = dlopen(library_filename.c_str(), RTLD_NOW | (global_symbols ? RTLD_GLOBAL : RTLD_LOCAL));
    char* error_str = dlerror();
    if (!*handle) {
      return common::Status(common::ONNXRUNTIME, common::FAIL,
                            "Failed to load library " + library_filename + " with error: " + error_str);
    }
    return common::Status::OK();
  }

  common::Status UnloadDynamicLibrary(void* handle) const override {
    if (!handle) {
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Got null library handle");
    }
    dlerror();  // clear any old error_str
    int retval = dlclose(handle);
    char* error_str = dlerror();
    if (retval != 0) {
      return common::Status(common::ONNXRUNTIME, common::FAIL,
                            "Failed to unload library with error: " + std::string(error_str));
    }
    return common::Status::OK();
  }

  common::Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override {
    dlerror();  // clear any old error str

    // search global space if handle is nullptr.
    // value of RTLD_DEFAULT differs across posix platforms (-2 on macos, 0 on linux).
    handle = handle ? handle : RTLD_DEFAULT;
    *symbol = dlsym(handle, symbol_name.c_str());

    char* error_str = dlerror();
    if (error_str) {
      return common::Status(common::ONNXRUNTIME, common::FAIL,
                            "Failed to get symbol " + symbol_name + " with error: " + error_str);
    }
    // it's possible to get a NULL symbol in our case when Schemas are not custom.
    return common::Status::OK();
  }

  std::string FormatLibraryFileName(const std::string& name, const std::string& version) const override {
    std::string filename;
    if (version.empty()) {
      filename = "lib" + name + ".so";
    } else {
      filename = "lib" + name + ".so" + "." + version;
    }
    return filename;
  }

  // \brief returns a provider that will handle telemetry on the current platform
  const Telemetry& GetTelemetryProvider() const override {
    return telemetry_provider_;
  }

  // \brief returns a value for the queried variable name (var_name)
  std::string GetEnvironmentVar(const std::string& var_name) const override {
    char* val = getenv(var_name.c_str());
    return val == NULL ? std::string() : std::string(val);
  }

 private:
  Telemetry telemetry_provider_;
#ifdef ORT_USE_CPUINFO
  PosixEnv() {
    cpuinfo_available_ = cpuinfo_initialize();
    if (!cpuinfo_available_) {
      LOGS_DEFAULT(INFO) << "cpuinfo_initialize failed";
    }
  }
  bool cpuinfo_available_{false};
#endif  // ORT_USE_CPUINFO
};

}  // namespace

// REGISTER_FILE_SYSTEM("", PosixFileSystem);
// REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);
Env& Env::Default() {
  return PosixEnv::Instance();
}

}  // namespace onnxruntime
