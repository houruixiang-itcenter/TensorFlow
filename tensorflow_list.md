- 1.运行tensorflow
- 2.人工神经网络简介
- 3.训练深度神经网络
- 4.跨设备和服务器的分布式TensorFlow
- 5.卷积神经网络
- 6.循环神经网络
- 7.自动编码器
- 8.强化学习


> 同样激活隔离环境  和Scikit-Learn一样
- cd $HOME/ml
- source env/bin/activate
- 安装TensorFlow:pip3 --default-timeout=100 install --upgrade tensorflow
error:pip._vendor.urllib3.exceptions.ReadTimeoutError: 
HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.


moudle : tensorflow-master  
- 
> tensorflow初识

- 计算图: clac_chart  <TensorFlow计算步骤>
- TensorFlow线性回归: core/reg_mode/tensorflow_reg 
- TensorFlow梯度下降: core/reg_mode/reg_gradient_descent
- TensorFlow保存状态节点: core/reg_mode/Saver
- TensorFlow计算图和节点模型线上: core/tebsor_board/tebsor_board
- TensorFlow命名空间 & 模块化: core/tebsor_board/tebsor_rules
- TensorFlow模块之间共享变量: core/share_variables/share_val
    
moudle : artificial_neural_network 
-
> 人工神经网络
- 单层感知器LTU atificial_neural_network/core/LTU
- 多层感知器MLP atificial_neural_network/core/MLP -- DNNClassifier
- 纯tensorflow实现多层感知器的深刻网络
1. atificial_neural_network/DNN 下所有py file


moudle : deep_neural_network 
-
- 实现批量归一化(零中心化,归一化  针对input) deep_neural_network/performance_evaluation/batch_normallization
- 梯度裁剪 deep_neural_network/performance_evaluation/batch_gradient_tailoring
- 学习效率调度(优化学习效率) deep_neural_network/optimization/learning_rate_optimization
- l1 & l2 范数正则化 deep_neural_network/optimization/regularization_optimization
- dropout正则化 deep_neural_network/optimization/dropout_optimization
- 最大范数正则化 deep_neural_network/optimization/max_regularization_optimization


MAC OX下安装tensorflow-gpu
1. 先根据macOS 10.12.6版本，在这个网站http://www.macvidcards.com/drivers.html找到相应的GPU Driver
我的mac是(16G1314) 378.05.05.25f07 

2.再进入CUDA Driver页面http://www.nvidia.com/object/mac-driver-archive.html，
找到和GPU driver匹配的版本(cudadriver-8.0.90-macos笔者下的是这个)，然后下载安装。
我的是这个:cudadriver-8.0.90-macos

3.再进入CUDA Toolkit页面https://developer.nvidia.com/cuda-toolkit-archive，
下载、安装CUDA Toolkit（笔者安装的是cuda_8.0.61_mac） 

4.CUDNN，笔者这里下的是 cudnn-8.0-osx-x64-v6.0
 https://developer.nvidia.com/rdp/cudnn-download 
 
5.配置1 & 2 & 3 的环境



moudle : convolutional_neural_network 
-
卷积神经网络


method_of_optimizinge-network
- 
优化网络的方法 
 