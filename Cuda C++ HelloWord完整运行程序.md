### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(hellocuda LANGUAGES CXX CUDA)

add_executable(main main.cu)
```

### main.cu

```c
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>


__global__ void kernel() {
  printf("hello cuda\n");
}

int main() {
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
```

### run.sh

```shell
#!/bin/bash

# 清理之前的构建文件（可选）
rm -rf build
mkdir build
cd build

# 运行CMake生成构建文件
cmake ..

# 编译程序
make

# 检查编译是否成功
if [ $? -eq 0 ]; then
  echo "编译成功"

  # 运行程序
  ./main
  echo "运行成功"
else
  echo "编译失败"
fi
```

### 文件结构

```shell
root@CD-DZ0104843:/home/hanbabang/2_workspace/cuda# tree
.
├── build
│   ├── CMakeCache.txt
│   ├── CMakeFiles
│   │   ├── 3.26.4
│   │   │   ├── CMakeCUDACompiler.cmake
│   │   │   ├── CMakeCXXCompiler.cmake
│   │   │   ├── CMakeDetermineCompilerABI_CUDA.bin
│   │   │   ├── CMakeDetermineCompilerABI_CXX.bin
│   │   │   ├── CMakeSystem.cmake
│   │   │   ├── CompilerIdCUDA
│   │   │   │   ├── a.out
│   │   │   │   ├── CMakeCUDACompilerId.cu
│   │   │   │   └── tmp
│   │   │   │       ├── a_dlink.fatbin
│   │   │   │       ├── a_dlink.fatbin.c
│   │   │   │       ├── a_dlink.o
│   │   │   │       ├── a_dlink.reg.c
│   │   │   │       ├── a_dlink.sm_52.cubin
│   │   │   │       ├── CMakeCUDACompilerId.cpp1.ii
│   │   │   │       ├── CMakeCUDACompilerId.cpp4.ii
│   │   │   │       ├── CMakeCUDACompilerId.cudafe1.c
│   │   │   │       ├── CMakeCUDACompilerId.cudafe1.cpp
│   │   │   │       ├── CMakeCUDACompilerId.cudafe1.gpu
│   │   │   │       ├── CMakeCUDACompilerId.cudafe1.stub.c
│   │   │   │       ├── CMakeCUDACompilerId.fatbin
│   │   │   │       ├── CMakeCUDACompilerId.fatbin.c
│   │   │   │       ├── CMakeCUDACompilerId.module_id
│   │   │   │       ├── CMakeCUDACompilerId.o
│   │   │   │       ├── CMakeCUDACompilerId.ptx
│   │   │   │       └── CMakeCUDACompilerId.sm_52.cubin
│   │   │   └── CompilerIdCXX
│   │   │       ├── a.out
│   │   │       ├── CMakeCXXCompilerId.cpp
│   │   │       └── tmp
│   │   ├── cmake.check_cache
│   │   ├── CMakeConfigureLog.yaml
│   │   ├── CMakeDirectoryInformation.cmake
│   │   ├── CMakeScratch
│   │   ├── main.dir
│   │   │   ├── build.make
│   │   │   ├── cmake_clean.cmake
│   │   │   ├── compiler_depend.make
│   │   │   ├── compiler_depend.ts
│   │   │   ├── DependInfo.cmake
│   │   │   ├── depend.make
│   │   │   ├── flags.make
│   │   │   ├── linkLibs.rsp
│   │   │   ├── link.txt
│   │   │   ├── main.cu.o
│   │   │   ├── main.cu.o.d
│   │   │   ├── objects1.rsp
│   │   │   └── progress.make
│   │   ├── Makefile2
│   │   ├── Makefile.cmake
│   │   ├── pkgRedirects
│   │   ├── progress.marks
│   │   └── TargetDirectories.txt
│   ├── cmake_install.cmake
│   ├── main
│   └── Makefile
├── CMakeLists.txt
├── executable
├── main.cu
└── run.sh
```

### 执行程序

```shell
root@CD-DZ0104843:/home/hanbabang/2_workspace/cuda# ./run.sh 
-- The CXX compiler identification is GNU 7.5.0
-- The CUDA compiler identification is NVIDIA 12.1.105
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Configuring done (2.3s)
CMake Warning (dev) in CMakeLists.txt:
  Policy CMP0104 is not set: CMAKE_CUDA_ARCHITECTURES now detected for NVCC,
  empty CUDA_ARCHITECTURES not allowed.  Run "cmake --help-policy CMP0104"
  for policy details.  Use the cmake_policy command to set the policy and
  suppress this warning.

  CUDA_ARCHITECTURES is empty for target "main".
This warning is for project developers.  Use -Wno-dev to suppress it.

CMake Warning (dev) in CMakeLists.txt:
  Policy CMP0104 is not set: CMAKE_CUDA_ARCHITECTURES now detected for NVCC,
  empty CUDA_ARCHITECTURES not allowed.  Run "cmake --help-policy CMP0104"
  for policy details.  Use the cmake_policy command to set the policy and
  suppress this warning.

  CUDA_ARCHITECTURES is empty for target "main".
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Generating done (0.0s)
-- Build files have been written to: /home/hanbabang/2_workspace/cuda/build
[ 50%] Building CUDA object CMakeFiles/main.dir/main.cu.o
[100%] Linking CUDA executable main
[100%] Built target main
编译成功
hello cuda
运行成功
```

