### 1 查询Nvidia GPU相关信息

CMD输入nvidia-smi查询GPU相关信息，注意到显卡驱动为535.98，CUDA Version为12.2。因为后续要下载CUDA TOOL KIT，目前官网为CUDA12，所以nvidia-smi查询的信息，CUDA Version如果低于12，则需要更新显卡驱动。

![image-20230608190518373](https://hanbabang-1311741789.cos.ap-chengdu.myqcloud.com/Pics/image-20230608190518373.png)

### 2 更新显卡驱动（若有必要）

以ThinkPad T14为例，其显卡为Nvidia Geforce MX330。进入Nvidia官网[官方驱动 | NVIDIA](https://www.nvidia.cn/Download/index.aspx?lang=cn)查询并下载最新显卡驱动安装。更新后，CUDA Version达到12.2，如上图。

![image-20230608190812937](https://hanbabang-1311741789.cos.ap-chengdu.myqcloud.com/Pics/image-20230608190812937.png)

![image-20230608190736308](https://hanbabang-1311741789.cos.ap-chengdu.myqcloud.com/Pics/image-20230608190736308.png)

### 3 下载安装CUDA Toolkit

以WSL2的Ubuntu为例，进入官网[CUDA Toolkit 12.1 Update 1 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)选择以下选项。

![image-20230608191028856](https://hanbabang-1311741789.cos.ap-chengdu.myqcloud.com/Pics/image-20230608191028856.png)

根据官网安装指引，逐条执行以下指令安装CUDA Toolkit。

![image-20230608191147424](https://hanbabang-1311741789.cos.ap-chengdu.myqcloud.com/Pics/image-20230608191147424.png)

安装好后，在/usr/local/cuda/bin中可找到nvcc，可将其添加到环境变量~/.bashrc。

![image-20230608191509064](https://hanbabang-1311741789.cos.ap-chengdu.myqcloud.com/Pics/image-20230608191509064.png)

![image-20230608191616230](https://hanbabang-1311741789.cos.ap-chengdu.myqcloud.com/Pics/image-20230608191616230.png)