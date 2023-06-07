### GPUMatrixMulti.cc

```cpp
#include <dlfcn.h>
#include <stdio.h>

#include <fstream>
#include <sstream>

#include "CL/cl.h"

cl_context CreateContext(void *handle) {
  cl_context context = nullptr;
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;

  // First, select an OpenCL platform to run on.
  // For this example, simply choose the first available platform.
  // Normally, you woudle query for all available platform and select the most
  // appropriate one.
  typedef cl_int (*m_pfnCLGetPlatformIDs)(
      cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms);
  m_pfnCLGetPlatformIDs getPlatformIDs =
      (m_pfnCLGetPlatformIDs)dlsym(handle, "clGetPlatformIDs");
  errNum = getPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if (errNum != CL_SUCCESS || numPlatforms < 0) {
    printf("Failed to find any OpenCL platforms. \n");
    return NULL;
  }

  // Next, create an OpenCL context on the platform.
  // Attempt to create a GPU-based context, and if that fails, try to create
  // a CPU-based context.
  cl_context_properties contextProperties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0};
  typedef cl_context (*m_pfCLCreateContextFromType)(
      const cl_context_properties *properties, cl_device_type device_type,
      void(CL_CALLBACK * pfn_notify)(const char *errinfo,
                                     const void *private_info, size_t cb,
                                     void *user_data),
      void *user_data, cl_int *errcode_ret);
  m_pfCLCreateContextFromType createContextFromType =
      (m_pfCLCreateContextFromType)dlsym(handle, "clCreateContextFromType");
  context = createContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL,
                                  NULL, &errNum);
  if (errNum != CL_SUCCESS) {
    printf("Could not create GPU context, trying CPU... \n");
    context = createContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL,
                                    NULL, &errNum);
    if (errNum != CL_SUCCESS) {
      printf("Failed to create an OpenCL GPU or CPU context... \n");
      return NULL;
    }
  }
  printf("Succesed to create OpenCL context! \n");
  return context;
}

cl_command_queue CreateCommandQueue(void *handle, const cl_context context,
                                    cl_device_id *device) {
  cl_command_queue commandQueue = NULL;
  cl_int errNum;
  cl_device_id *devices = nullptr;
  size_t deviceBufferSize = -1;

  // First get the size of the devices buffer
  typedef cl_int (*m_pfCLGetContextInfo)(
      cl_context context, cl_context_info param_name, size_t param_value_size,
      void *param_value, size_t *param_value_size_ret);
  m_pfCLGetContextInfo getContextInfo =
      (m_pfCLGetContextInfo)(dlsym(handle, "clGetContextInfo"));
  errNum =
      getContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  if (errNum != CL_SUCCESS) {
    printf("Failed call to clGetContextInfo... \n");
    return NULL;
  }
  if (deviceBufferSize <= 0) {
    printf("No devices available... \n");
    return NULL;
  }
  // Allocate memory for the devices buffer
  devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = getContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize,
                          devices, NULL);
  if (errNum != CL_SUCCESS) {
    printf("Failed to get device IDs... \n");
    return NULL;
  }

  // In this example, just choose the first available device.
  // In a real program, would likely all available devices or choose the
  // highest performance device based on OpenCL device queries.
  typedef cl_command_queue (*m_pfcCLCreateCommandQueue)(
      cl_context context, cl_device_id device,
      cl_command_queue_properties properties, cl_int * errcode_ret);
  m_pfcCLCreateCommandQueue createCommandQueue =
      (m_pfcCLCreateCommandQueue)(dlsym(handle, "clCreateCommandQueue"));
  commandQueue = createCommandQueue(context, devices[0], 0, NULL);
  if (commandQueue == NULL) {
    printf("Failed to create commandQueue for device 0 \n");
    return NULL;
  }
  printf("Succesed to create OpenCL commandQueue! \n");
  *device = devices[0];
  delete[] devices;
  return commandQueue;
}

cl_program CreateProgram(void *handle, cl_context context, const char *fileName,
                         cl_device_id device) {
  cl_int errNum;
  cl_program program;
  std::ifstream kernelFile(fileName, std::ios::in);
  if (!kernelFile.is_open()) {
    printf("Failed to open for reading: %s \n", fileName);
    return NULL;
  }
  std::ostringstream oss;
  oss << kernelFile.rdbuf();
  std::string strStdstr = oss.str();
  const char *srcStr = strStdstr.c_str();

  typedef cl_program (*m_pfCLCreateProgramWithSource)(
      cl_context context, cl_uint count, const char **strings,
      const size_t *lengths, cl_int *errcode_ret);
  m_pfCLCreateProgramWithSource createProgramWithSource =
      (m_pfCLCreateProgramWithSource)(dlsym(handle,
                                            "clCreateProgramWithSource"));
  program =
      createProgramWithSource(context, 1, (const char **)&srcStr, NULL, NULL);
  if (program == NULL) {
    printf("Failed to create OpenCL program for source. \n");
    return NULL;
  }
  typedef cl_int (*m_pfCLBuildProgram)(
      cl_program program, cl_uint num_devices, const cl_device_id *device_list,
      const char *options,
      void(CL_CALLBACK * pfn_notify)(cl_program program, void *user_data),
      void *user_data);
  m_pfCLBuildProgram buildProgram =
      (m_pfCLBuildProgram)(dlsym(handle, "clBuildProgram"));
  errNum = buildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    char buildLog[16384];
    typedef cl_int (*m_pfCLGetProgramBuildInfo)(
        cl_program program, cl_device_id device,
        cl_program_build_info param_name, size_t param_value_size,
        void *param_value, size_t *param_value_size_ret);
    m_pfCLGetProgramBuildInfo getProgramBuildInfo =
        (m_pfCLGetProgramBuildInfo)(dlsym(handle, "clGetProgramBuildInfo"));
    getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog),
                        buildLog, NULL);
    printf("Error in kernel: %s \n", buildLog);
    // release program
    typedef cl_int (*m_pfCLReleaseProgram)(cl_program program);
    m_pfCLReleaseProgram releaseProgram =
        (m_pfCLReleaseProgram)(dlsym(handle, "clReleaseProgram"));
    releaseProgram(program);
    return NULL;
  }
  printf("Succesed to build OpenCL program! \n");
  return program;
}

cl_kernel CreateKernel(void *handle, cl_program program) {
  cl_kernel kernel = NULL;
  typedef cl_kernel (*m_pfCLCreateKernel)(
      cl_program program, const char *kernel_name, cl_int *errcode_ret);
  m_pfCLCreateKernel createKernel =
      (m_pfCLCreateKernel)(dlsym(handle, "clCreateKernel"));
  kernel = createKernel(program, "matrix_multi", NULL);
  if (kernel == NULL) {
    printf("Failed to create kernel. \n");
    return NULL;
  }
  printf("Succesed to create OpenCL kernel! \n");
  return kernel;
}

bool CreateMemObjects(void *handle, cl_context context, cl_mem memObjects[3],
                      int array_size, float *a, float *b) {
  typedef cl_mem (*m_pfCLCreateBuffer)(cl_context context, cl_mem_flags flags,
                                       size_t size, void *host_ptr,
                                       cl_int *errcode_ret);
  m_pfCLCreateBuffer createBuffer =
      (m_pfCLCreateBuffer)(dlsym(handle, "clCreateBuffer"));
  memObjects[0] = createBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * array_size, a, NULL);
  memObjects[1] = createBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * array_size, b, NULL);
  memObjects[2] = createBuffer(context, CL_MEM_READ_WRITE,
                               sizeof(float) * array_size, NULL, NULL);
  if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL) {
    printf("Error creating memory objects. \n");
    return false;
  }
  printf("Succesed to create memory objects! \n");
  return true;
}

cl_int SetKernelArg(void *handle, cl_kernel kernel, cl_mem memObjects[3]) {
  cl_int errNum;
  typedef cl_int (*m_pfCLSetKernelArg)(cl_kernel kernel, cl_uint arg_index,
                                       size_t arg_size, const void *arg_value);
  m_pfCLSetKernelArg setKernelArg =
      (m_pfCLSetKernelArg)(dlsym(handle, "clSetKernelArg"));
  errNum = setKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
  errNum |= setKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
  errNum |= setKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
  if (errNum != CL_SUCCESS) {
    printf("Error settings the kernel arguments. \n");
    return -1;
  }
  printf("Succesed to set kernel arguments! \n");
  return errNum;
}

cl_int EnqueueKernel(void *handle, cl_command_queue commandQueue,
                     cl_kernel kernel, size_t globalWorkSize[1],
                     size_t localWorkSize[1]) {
  cl_int errNum;
  typedef cl_int (*m_pfCLEnqueueNDRangeKernel)(
      cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
      const size_t *global_work_offset, const size_t *global_work_size,
      const size_t *local_work_size, cl_uint num_events_in_wait_list,
      const cl_event *event_wait_list, cl_event *event);
  m_pfCLEnqueueNDRangeKernel enqueueNDRangeKernel =
      (m_pfCLEnqueueNDRangeKernel)(dlsym(handle, "clEnqueueNDRangeKernel"));
  errNum = enqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize,
                                localWorkSize, 0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    printf("Error queuing kernel for execution.");
    return -1;
  }
  printf("Succesed to executed kernel! \n");
  return errNum;
}

cl_int ReadOutputBuffer(void *handle, cl_command_queue commandQueue,
                        cl_mem memObjects[3], int array_size, float *result) {
  cl_int errNum;
  typedef cl_int (*m_pfCLEnqueueReadBuffer)(
      cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
      size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list,
      const cl_event *event_wait_list, cl_event *event);
  m_pfCLEnqueueReadBuffer enqueueReadBuffer =
      (m_pfCLEnqueueReadBuffer)(dlsym(handle, "clEnqueueReadBuffer"));
  errNum = enqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                             array_size * sizeof(float), result, 0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    printf("Error reading result buffer. \n");
    return -1;
  }
  printf("Succesed to read output buffer! \n");
  return errNum;
}

void Release(void *handle, cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memobj[3]) {
  if (NULL != context) {
    // release context
    typedef cl_int (*m_pfCLReleaseContext)(cl_context context);
    m_pfCLReleaseContext releaseContext =
        (m_pfCLReleaseContext)dlsym(handle, "clReleaseContext");
    releaseContext(context);
  }

  if (NULL != commandQueue) {
    // release command-queue
    typedef cl_int (*m_pfCLReleaseCommandQueue)(cl_command_queue command_queue);
    m_pfCLReleaseCommandQueue releaseCommandQueue =
        (m_pfCLReleaseCommandQueue)(dlsym(handle, "clReleaseCommandQueue"));
    releaseCommandQueue(commandQueue);
  }

  if (NULL != program) {
    // release program
    typedef cl_int (*m_pfCLReleaseProgram)(cl_program program);
    m_pfCLReleaseProgram releaseProgram =
        (m_pfCLReleaseProgram)(dlsym(handle, "clReleaseProgram"));
    releaseProgram(program);
  }

  if (NULL != kernel) {
    // release kernel
    typedef cl_int (*m_pfCLReleaseKernel)(cl_kernel kernel);
    m_pfCLReleaseKernel releaseKernel =
        (m_pfCLReleaseKernel)(dlsym(handle, "clReleaseKernel"));
    releaseKernel(kernel);
  }

  // release memory objects
  typedef cl_int (*m_pfCLReleaseMemObject)(cl_mem memobj);
  m_pfCLReleaseMemObject releaseMemObject =
      (m_pfCLReleaseMemObject)(dlsym(handle, "clReleaseMemObject"));
  releaseMemObject(memobj[0]);
  releaseMemObject(memobj[1]);
  releaseMemObject(memobj[2]);
}

int main() {
  void *handle = dlopen("/vendor/lib64/libOpenCL.so", RTLD_LAZY);
  if (handle == NULL) {
    printf("ERROR:%s:dlopen failed! \n", dlerror());
    return -1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  cl_context context = 0;
  cl_command_queue commandQueue = 0;
  cl_program program = 0;
  cl_device_id device = 0;
  cl_mem memObjects[3] = {0, 0, 0};
  cl_int errNum;
  cl_kernel kernel = nullptr;

  // create an OpenCL context on first available platform.
  context = CreateContext(handle);
  if (NULL == context) {
    printf("Failed to create OpenCL context. \n");
    return -1;
  }

  // Create a command-queue on the first device available
  // on the created context.
  commandQueue = CreateCommandQueue(handle, context, &device);
  if (NULL == commandQueue) {
    printf("Failed to create commandQueue for device 0 \n");
    Release(handle, context, commandQueue, program, kernel, memObjects);
    return -1;
  }

  // Create OpenCL program from GPUMatrixMulti.cl kernel source.
  const char *fileName = "/data/local/tmp/MatrixMulti.cl";
  program = CreateProgram(handle, context, fileName, device);
  if (NULL == program) {
    printf("Failed to create OpenCL program for source. \n");
    Release(handle, context, commandQueue, program, kernel, memObjects);
    return -1;
  }

  // Create OpenCL kernel
  kernel = CreateKernel(handle, program);
  if (NULL == kernel) {
    printf("Failed to create kernel. \n");
    Release(handle, context, commandQueue, program, kernel, memObjects);
    return -1;
  }

  const int ARRAY_SIZE = 1024;
  float result[ARRAY_SIZE];
  float a[ARRAY_SIZE];
  float b[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) {
    a[i] = i;
    a[i] = i * 2;
  }
  if (!CreateMemObjects(handle, context, memObjects, ARRAY_SIZE, a, b)) {
    Release(handle, context, commandQueue, program, kernel, memObjects);
    return -1;
  }

  // Create the Kernel arguments (a,b,result)
  errNum = SetKernelArg(handle, kernel, memObjects);
  if (errNum != CL_SUCCESS) {
    printf("Error settings the kernel arguments. \n");
    Release(handle, context, commandQueue, program, kernel, memObjects);
    return -1;
  }

  // Set Group Size
  size_t globalWorkSize[1] = {ARRAY_SIZE};
  size_t localWorkSize[1] = {1};

  // Queue the kernel up for execution across the array.
  errNum = EnqueueKernel(handle, commandQueue, kernel, globalWorkSize,
                         localWorkSize);
  if (errNum != CL_SUCCESS) {
    printf("Error queuing kernel for execution.");
    Release(handle, context, commandQueue, program, kernel, memObjects);
    return -1;
  }

  // Read the output buffer back to he host.
  errNum =
      ReadOutputBuffer(handle, commandQueue, memObjects, ARRAY_SIZE, result);
  if (errNum != CL_SUCCESS) {
    printf("Error reading result buffer. \n");
    Release(handle, context, commandQueue, program, kernel, memObjects);
    return -1;
  }

  for (int i = 0; i < ARRAY_SIZE; i++) {
    printf("result: %.3f \n", result[i]);
  }

  // Release
  Release(handle, context, commandQueue, program, kernel, memObjects);

  auto end = std::chrono::high_resolution_clock::now();
  auto cost_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  printf("====> OpenCL cost time: %.5fs\n", cost_time.count());

  if (NULL != handle) {
    dlclose(handle);
    handle = NULL;
  }
  return 0;
}
```

### MatrixMulti.cl

```cc
// OpenCL Kernel Function
__kernel void matrix_multi(__global const float* A, 
                           __global const float* B,
                           __global float* result)
                           {
                             int i = get_global_id(0);
                             int j = get_global_id(1);
                             if((i < 1024) && (j < 1024)) {
                               for(int k = 0; k < 1024; k++) {
                                 result[i*1024 + j] += A[i*1024 + k] * B[k*1024 + j];
                               }
                             }
                           }


```

