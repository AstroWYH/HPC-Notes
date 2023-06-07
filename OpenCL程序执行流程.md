# 导读

1. OpenCL概念基础
2. OpenCL名词介绍
3. OpenCL的编程步骤
4. 程序示例，矩阵乘法
5. OpenCL程序执行流程
6. 选择OpenCL平台并创建一个上下文
7. 选择设备并创建命令队列
8. 创建和构建程序对象
9. 创建内核
10. 创建内存对象
11. 设置内核参数
12. 设置Group Size
13. 执行内核
14. 读取结果
15. 释放资源
16. OpenCL内核函数限制
17. 示例程序源文件

# 正文

OpenCL是面向由CPU、GPU和其它处理器组合构成的计算机进行编程的行业标准框架。

它由一门用于编写kernels （在OpenCL设备上运行的函数）的语言（基于C99）和一组用于定义并控制平台的API组成。

OpenCL提供了两种层面的并行机制：**任务并行**与**数据并行**。

## OpenCL概念基础

OpenCL支持大量不同类型的应用，面向异构平台的应用都必须完成以下步骤:

1. 发现构成异构系统的组件;
2. 探查这些组件的特征，使软件能够适应不同硬件单元的特定特性；
3. 创建将在平台上运行的指令块（内核）；
4. 建立并管理计算机中设计的内存对象；
5. 在系统中正确的组件上按正确的顺序执行内核；
6. 收集最终结果。

这些步骤通过OpenCL中一系列的API再加上一个面向内核的编程环境来完成。以上工作可以分为以下几种模型：

- **平台模型**(platform model): 异构系统的高层描述。
- **执行模型**(execution model):指令在异构平台上执行的抽象表示。
- **内存模型**(memory model): OpenCL中的内存区域集合以及一个OpenCL计算期间这些内存区域如何交互。
- **编程模型**(programming model): 程序员设计算法莱实现一个应用时的高层抽象。

## OpenCL名词介绍

一个完整的OpenCL加速技术过程涉及到平台（Platform）、设备(Device)、上下文(Context)、OpenCL程序（Program）、

指令队列（Command）、核函数（Kernel）、内存对象（Memory Object）、调用设备接口（NDRange）。

- **Platform(平台)**:主机加上OpenCL框架管理下的若干设备构成了这个平台，通过这个平台，应用程序可以与设备共享资源并在设备上执行kernel。

​                       实际使用中基本上一个厂商对应一个Platform，比如Intel, AMD都是这样。

- **Device（设备）**：官方的解释是计算单元（Compute Units）的集合。CPU和GPU是典型的device。
- **Context（上下文）**：OpenCL的Platform上共享和使用资源的环境，包括kernel、device、memory objects、command queue等。使用中一般一个Platform对应一个Context。
- **Program**：OpenCL程序，由kernel函数、其他函数和声明等组成。
- **Command Queue（指令队列）**：在指定设备上管理多个指令（Command）。队列里指令执行可以顺序也可以乱序。一个设备可以对应多个指令队列。
- **NDRange**：主机端运行设备端kernel函数的主要接口。实际上还有其他的，NDRange是非常常见的，用于分组运算。（ND-Rang以N维网格形式组织的，N=1，2或3）
- **Kernel（核函数）**：可以从主机端调用，运行在设备端的函数。
- **Memory Object（内存对象）**：在主机和设备之间传递数据的对象，一般映射到OpenCL程序中的global memory。有两种具体的类型：Buffer Object（缓存对象）和Image Object（图像对象）。

## OpenCL的编程步骤

1. Discover and initialize the platforms(调用两次clGetPlatformIDs函数，第一次获取可用的平台数量，第二次获取一个可用的平台。)
2. Discover and initialize the devices(调用两次clGetDeviceIDs函数，第一次获取可用的设备数量，第二次获取一个可用的设备。)
3. Create a context(调用clCreateContext函数创建上下文, context可能会管理多个设备device。)
4. Create a command queue(调用clCreateCommandQueue函数，一个设备device对应一个command queue。上下文conetxt将命令发送到设备对应的command queue，设备就可以执行命令队列里的命令。）
5. Create device buffers(调用clCreateBuffer函数）
   1. Buffer中保存的是数据对象，就是设备执行程序需要的数据保存在其中。
   2. Buffer由上下文conetxt创建，这样上下文管理的多个设备就会共享Buffer中的数据。
6. Write host data to device buffers(调用clEnqueueWriteBuffer函数）
7. Create and compile the program(创建程序对象，程序对象就代表你的程序源文件或者二进制代码数据。)
8. Create the kernel(调用clCreateKernel函数，根据程序对象，生成kernel对象，表示设备程序的入口。）
9. Set the kernel arguments(调用clSetKernelArg函数）
10. Configure the work-item structure(设置worksize，配置work-item的组织形式（维数，group组成等)）
11. Enqueue the kernel for execution(调用clEnqueueNDRangeKernel函数,将kernel对象，以及 work-item参数放入命令队列中进行执行。）
12. Read the output buffer back to the host(调用clEnqueueReadBuffer函数）
13. Release OpenCL resources（至此结束整个运行过程）

![image-20230607172726143](https://hanbabang-1311741789.cos.ap-chengdu.myqcloud.com/Pics/image-20230607172726143.png)

## 程序示例，矩阵乘法

**C实现**

```cc
const int n = 1024;
void MatMult(int n, const float* A, const float* B, float* C) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      for(int k = 0; k < n; k++) {
        C[i*n + j] += A[i*n + k] * B[k*n + j];
      }
    }
  }
}
 
Platform: SDM8250+，CPU cost time: 9.50071 s
```

**内核函数实现**

```cc
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
 
Platform: SDM8250+，OpenCL cost time: 0.08117 s
```

以上，两个1024 x 1024的矩阵相乘，在CPU上为串行执行，在GPU上，对所有的乘法可以进行并行，

每个线程计算一个元素的方法来代替CPU程序中的循环计算，两个版本在性能上差距约为117倍。

## OpenCL程序执行流程

### 选择OpenCL平台并创建一个上下文

创建OpenCL程序的第一步是创建一个上下文。首先调用clGetPlatformIDs来获取第一个可用的平台，得到第一个可用平台的cl_platform_id之后，

再调用clCreateContextFromType创建一个上下文，这个调用会尝试为一个GPU设备创建一个上下文，如果尝试失败，程序会做下一个尝试，

为CPU设备创建一个上下文。

```cc
cl_context CreateContext(void *handle)
{
  cl_context context = nullptr;
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
 
  // First, select an OpenCL platform to run on.
  // For this example, simply choose the first available platform.
  // Normally, you woudle query for all available platform and select the most appropriate one.
  typedef cl_int (*m_pfnCLGetPlatformIDs)(cl_uint num_entries,
                                          cl_platform_id * platforms,
                                          cl_uint * num_platforms);
  m_pfnCLGetPlatformIDs getPlatformIDs = (m_pfnCLGetPlatformIDs)dlsym(handle, "clGetPlatformIDs");
  errNum = getPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if (errNum != CL_SUCCESS || numPlatforms < 0)
  {
    printf("Failed to find any OpenCL platforms. \n");
    return NULL;
  }
 
  // Next, create an OpenCL context on the platform.
  // Attempt to create a GPU-based context, and if that fails, try to create
  // a CPU-based context.
  cl_context_properties contextProperties[] = {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)firstPlatformId,
      0};
  typedef cl_context (*m_pfCLCreateContextFromType)(const cl_context_properties *properties,
                                                    cl_device_type device_type,
                                                    void(CL_CALLBACK * pfn_notify)(const char *errinfo,
                                                                                   const void *private_info,
                                                                                   size_t cb,
                                                                                   void *user_data),
                                                    void *user_data,
                                                    cl_int *errcode_ret);
  m_pfCLCreateContextFromType createContextFromType = (m_pfCLCreateContextFromType)dlsym(handle, "clCreateContextFromType");
  context = createContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
  if (errNum != CL_SUCCESS)
  {
    printf("Could not create GPU context, trying CPU... \n");
    context = createContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
      printf("Failed to create an OpenCL GPU or CPU context... \n");
      return NULL;
    }
  }
  printf("Succesed to create OpenCL context! \n");
  return context;
}
```

### 选择设备并创建命令队列

设备在计算机硬件底层，如GPU或CPU。要与设备通信，应用程序必须为它创建一个命令队列。将在设备上完成的操作要在命令队列中排队。

```cc
cl_command_queue CreateCommandQueue(void *handle, const cl_context context, cl_device_id *device){
  cl_command_queue commandQueue = NULL;
  cl_int errNum;
  cl_device_id *devices = nullptr;
  size_t deviceBufferSize = -1;
   
  // First get the size of the devices buffer
  typedef cl_int (*m_pfCLGetContextInfo)(cl_context context,
                                 cl_context_info param_name,
                                 size_t param_value_size,
                                 void *param_value,
                                 size_t *param_value_size_ret);
  m_pfCLGetContextInfo getContextInfo = (m_pfCLGetContextInfo)(dlsym(handle, "clGetContextInfo"));
  errNum = getContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  if (errNum != CL_SUCCESS)
  {
    printf("Failed call to clGetContextInfo... \n");
    return NULL;
  }
  if (deviceBufferSize <= 0)
  {
    printf("No devices available... \n");
    return NULL;
  }
  // Allocate memory for the devices buffer
  devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = getContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
  if (errNum != CL_SUCCESS)
  {
    printf("Failed to get device IDs... \n");
    return NULL;
  }
 
  // In this example, just choose the first available device.
  // In a real program, would likely all available devices or choose the
  // highest performance device based on OpenCL device queries.
  typedef cl_command_queue (*m_pfcCLCreateCommandQueue)(cl_context context,
                                                cl_device_id device,
                                                cl_command_queue_properties properties,
                                                cl_int * errcode_ret);
  m_pfcCLCreateCommandQueue createCommandQueue = (m_pfcCLCreateCommandQueue)(dlsym(handle, "clCreateCommandQueue"));
  commandQueue = createCommandQueue(context, devices[0], 0, NULL);
  if (commandQueue == NULL)
  {
    printf("Failed to create commandQueue for device 0 \n");
    return NULL;
  }
  printf("Succesed to create OpenCL commandQueue! \n");
  *device = devices[0];
  delete[] devices;
  return commandQueue;
}
```

第一个clGetContextInfo调用会查询上下文的信息，得到存储上下文中所有可用设备ID所需要的缓冲区大小。这个大小将用来分配一个缓冲区，用于存储设备ID，

另一个clGetContextInfo调用则获取上下文中所有可用的设备。选择了所用设备之后，应用调用clCreateCommandQueue在所选择的设备上创建一个名列队列。

这个名列队列将用于将程序中要执行的内核排队，并读回其结果。

### 创建和构建程序对象

下一步将从.cl文件加载OpenCL C 内核源代码，由它创建一个程序对象，这个程序对象由内核源代码加载，然后进行编译，从而在与上下文的关联的设备上执行。

```cc
cl_program CreateProgram(void* handle, cl_context context, const char *fileName, cl_device_id device){
  cl_int errNum;
  cl_program program;
  std::ifstream kernelFile(fileName, std::ios::in);
  if (!kernelFile.is_open())
  {
    printf("Failed to open for reading: %s \n", fileName);
    return NULL;
  }
  std::ostringstream oss;
  oss << kernelFile.rdbuf();
  std::string strStdstr = oss.str();
  const char *srcStr = strStdstr.c_str();
 
  typedef cl_program (*m_pfCLCreateProgramWithSource)(cl_context context,
                                               cl_uint count,
                                               const char **strings,
                                               const size_t *lengths,
                                               cl_int *errcode_ret);
  m_pfCLCreateProgramWithSource createProgramWithSource = (m_pfCLCreateProgramWithSource)(dlsym(handle, "clCreateProgramWithSource"));
  program = createProgramWithSource(context, 1, (const char **)&srcStr, NULL, NULL);
  if (program == NULL)
  {
    printf("Failed to create OpenCL program for source. \n");
    return NULL;
  }
  typedef cl_int (*m_pfCLBuildProgram)(cl_program program,
                                cl_uint num_devices,
                                const cl_device_id *device_list,
                                const char *options,
                                void(CL_CALLBACK * pfn_notify)(cl_program program,
                                                               void *user_data),
                                void *user_data);
  m_pfCLBuildProgram buildProgram = (m_pfCLBuildProgram)(dlsym(handle, "clBuildProgram"));
  errNum = buildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS)
  {
    char buildLog[16384];
    typedef cl_int (*m_pfCLGetProgramBuildInfo)(cl_program program,
                                         cl_device_id device,
                                         cl_program_build_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret);
    m_pfCLGetProgramBuildInfo getProgramBuildInfo = (m_pfCLGetProgramBuildInfo)(dlsym(handle, "clGetProgramBuildInfo"));
    getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);                                                                              
    printf("Error in kernel: %s \n", buildLog);
    // release program
    typedef cl_int(*m_pfCLReleaseProgram)(cl_program program);
    m_pfCLReleaseProgram releaseProgram = (m_pfCLReleaseProgram)(dlsym(handle, "clReleaseProgram"));
    releaseProgram(program);   
    return NULL;
  }
  printf("Succesed to build OpenCL program! \n");
  return program;
}
```

首先从磁盘加载cl内核文件,并存储在一个字符串中。通过调用clCreateProgramWithSource创建程序对象。之后，通过调用clBuildProgram编译内核源代码。这个函数会为关联的设备编译内核，

如果编译成功，则把编译代码存储在程序对象中，如果编译失败，可以使用clGetProgramBuildInfo获取构建日志。日志中包含了OpenCL内核编译过程中生成的编译错误。

### 创建内核

```cc
cl_kernel CreateKernel(void* handle， cl_program program) {
  cl_kernel kernel = NULL;
  typedef cl_kernel (*m_pfCLCreateKernel)(cl_program program,
                                   const char *kernel_name,
                                   cl_int *errcode_ret);
  m_pfCLCreateKernel createKernel = (m_pfCLCreateKernel)(dlsym(handle, "clCreateKernel"));
  kernel = createKernel(program, "matrix_multi", NULL);
  if (kernel == NULL) {
    printf("Failed to create kernel. \n");
    return NULL;
  }
  printf("Succesed to create OpenCL kernel! \n");
  return kernel;
}
```

### 创建内存对象

要执行OpenCL计算内核，需要中内存中分配内核函数的参数，以便在OpenCL设备上访问。在宿主机内存中创建这些参数之后，调用CreateMemObjects会把这些数据复制到内存对象，然后传入内核。

```cc
bool CreateMemObjects(void* handle, cl_context context, cl_mem memObjects[3], int array_size, float *a, float *b){
  typedef cl_mem (*m_pfCLCreateBuffer)(cl_context context,
                                cl_mem_flags flags,
                                size_t size,
                                void *host_ptr,
                                cl_int *errcode_ret);
  m_pfCLCreateBuffer createBuffer = (m_pfCLCreateBuffer)(dlsym(handle, "clCreateBuffer"));
  memObjects[0] = createBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * array_size, a, NULL);
  memObjects[1] = createBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * array_size, b, NULL);
  memObjects[2] = createBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * array_size, NULL, NULL);
  if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
  {
    printf("Error creating memory objects. \n");
    return false;
  }
  printf("Succesed to create memory objects! \n");
  return true;
}
```

为各个数组分别调用clCreateBuffer来创建一个内存对象。内存对象分配在设备内存中，可以由内核函数直接访问。对于输入(a和b)，缓冲区CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR

内存类型来创建，这说明数组对内核是只读的，可以从宿主机内存复制到设备内存。数组本身作为参数传递到clCreateBuffer，这会将数组的内容复制到设备上为内存对象分配的存储空间中。

result数组类型为CL_MEM_READ_WRITE创建，这说明这个数组对内核是可读、可写的。

### 设置内核参数

内核函数的所有参数需要使用clSetKernelArg设置。这个函数的第二个参数是待设置参数的索引。

```cc
cl_int SetKernelArg(void* handle, cl_kernel kernel, cl_mem memObjects[3]) {
  cl_int errNum;
  typedef cl_int (*m_pfCLSetKernelArg)(cl_kernel kernel,
                                cl_uint arg_index,
                                size_t arg_size,
                                const void *arg_value);
  m_pfCLSetKernelArg setKernelArg = (m_pfCLSetKernelArg)(dlsym(handle, "clSetKernelArg"));
  errNum = setKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
  errNum |= setKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
  errNum |= setKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
  if (errNum != CL_SUCCESS)
  {
    printf("Error settings the kernel arguments. \n");
    return -1;
  }
  printf("Succesed to set kernel arguments! \n");
  return errNum;
}
```

### 设置Group Size

globalWorkSize和localWorkSize确定内核如何在设备上的多个处理单元间分布。

```cc
size_t globalWorkSize[1] = {ARRAY_SIZE};
size_t localWorkSize[1] = {1};
```

### 执行内核

内核执行会放在命令队列中，由设备消费。

```cc
cl_int EnqueueKernel(void* handle, cl_command_queue commandQueue, cl_kernel kernel,
                    size_t globalWorkSize[1], size_t localWorkSize[1]) {
  cl_int errNum;
  typedef cl_int (*m_pfCLEnqueueNDRangeKernel)(cl_command_queue command_queue,
                                        cl_kernel kernel,
                                        cl_uint work_dim,
                                        const size_t *global_work_offset,
                                        const size_t *global_work_size,
                                        const size_t *local_work_size,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event *event_wait_list,
                                        cl_event *event);
  m_pfCLEnqueueNDRangeKernel enqueueNDRangeKernel = (m_pfCLEnqueueNDRangeKernel)(dlsym(handle, "clEnqueueNDRangeKernel"));
  errNum = enqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if (errNum != CL_SUCCESS)
  {
    printf("Error queuing kernel for execution.");
    return -1;
  }
  printf("Succesed to executed kernel! \n");
  return errNum;
}
```

### 读取结果

```cc
cl_int ReadOutputBuffer(void* handle, cl_command_queue commandQueue, cl_mem memObjects[3], int array_size, float *result){
  cl_int errNum;
  typedef cl_int (*m_pfCLEnqueueReadBuffer)(cl_command_queue command_queue,
                                     cl_mem buffer,
                                     cl_bool blocking_read,
                                     size_t offset,
                                     size_t size,
                                     void *ptr,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event *event_wait_list,
                                     cl_event *event);
  m_pfCLEnqueueReadBuffer enqueueReadBuffer = (m_pfCLEnqueueReadBuffer)(dlsym(handle, "clEnqueueReadBuffer"));
  errNum = enqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, array_size * sizeof(float), result, 0, NULL, NULL);
  if (errNum != CL_SUCCESS)
  {
    printf("Error reading result buffer. \n");
    return -1;
  }
  printf("Succesed to read output buffer! \n");
  return errNum;
}
```

### 释放资源

```cc
void Release(void* handle, cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memobj[3]) {
    if (NULL != context) {
      // release context
      typedef cl_int (*m_pfCLReleaseContext)(cl_context context);
      m_pfCLReleaseContext releaseContext = (m_pfCLReleaseContext)dlsym(handle, "clReleaseContext");
      releaseContext(context);
    }
 
    if (NULL != commandQueue) {
      // release command-queue
      typedef cl_int (*m_pfCLReleaseCommandQueue)(cl_command_queue command_queue);
      m_pfCLReleaseCommandQueue releaseCommandQueue = (m_pfCLReleaseCommandQueue)(dlsym(handle, "clReleaseCommandQueue"));
      releaseCommandQueue(commandQueue);
    }
 
    if (NULL != program) {
      // release program
      typedef cl_int(*m_pfCLReleaseProgram)(cl_program program);
      m_pfCLReleaseProgram releaseProgram = (m_pfCLReleaseProgram)(dlsym(handle, "clReleaseProgram"));
      releaseProgram(program);
    }
 
    if (NULL != kernel) {
      // release kernel
      typedef cl_int (*m_pfCLReleaseKernel)(cl_kernel kernel);
      m_pfCLReleaseKernel releaseKernel = (m_pfCLReleaseKernel)(dlsym(handle, "clReleaseKernel"));
      releaseKernel(kernel);
    }
 
    // release memory objects
    typedef cl_int(*m_pfCLReleaseMemObject)(cl_mem memobj);
    m_pfCLReleaseMemObject releaseMemObject = (m_pfCLReleaseMemObject)(dlsym(handle, "clReleaseMemObject"));
    releaseMemObject(memobj[0]);
    releaseMemObject(memobj[1]);
    releaseMemObject(memobj[2]);
}
```

## OpenCL内核函数限制

- 如果内核函数的参数是指针，则必须使用global、constant或local限定符；
- 内核函数的参数不能声明为指向指针的指针；
- 内核函数的参数不能用以下内置类型声明：bool、half、size_t、ptrdiff_t、intptr_t、uintptr_t或event_t；
- 内核函数的返回类型必须是void；
- 如果内核函数的参数声明为结构(struct)，则不能传入OpenCL对象(如缓冲区、图像)作为结构的元素；