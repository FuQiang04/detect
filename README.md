项目所需额外文件：

1.需要在本目录下存放bert-base-chinese文件夹，下载地址https://hf-mirror.com/google-bert/bert-base-chinese/tree/main
，将里面的所有文件下载并打包成bert-base-chinese文件夹

2.需要在本目录下存放data文件夹，其中包含两个文件夹，分别为cyberbullying和non_cyberbullying文件夹，分别存放了恶意图片和非恶意图片

项目文件说明：

1.prompt_config.py存储了大模型的提示词

2.generate_descriptions.py是通过调用大模型api对data文件夹中的训练数据进行图像描述，并将结果存入data.csv

3.train.py是通过调用data.csv中的训练数据训练一个bert模型

项目运行：

运行generate_descriptions.py生成data.csv，运行train.py生成bert模型，运行api.py打开页面

温馨提示：

若用摄像头录制视频上传的话，请尽量做点动作，要不然大模型会将视频视作静态的图片
