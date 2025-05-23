## 使用之前
1. 使用 pip 安装一下环境和 [devtoolkit](src/resources/devtoolkit-0.1.0.2025032502-py3-none-any.whl) 库：（自己写的所以没法从网络安装，主要是用来记录日志的）
```(shell)
conda env create -f src/resources/environment.yml
conda activate helldivers2-audio-assistant
pip install src/resources/devtoolkit-0.1.0.2025032502-py3-none-any.whl

# 或者直接使用初始化脚本（推荐）
./initialization.ps1
```
## 获得训练集
1. 在战备字典文件 [cmd-dict.csv](src/resources/cmd-dict.csv) 文件中填写需要使用语音呼叫的战备，格式为{索引},{呼号},{指令序列},{内部代号}
2. 运行 [record.py](src/python/record.py) 程序自动录制指令片段（默认1秒，这是最佳实践，可以改但是需要改很多文件很麻烦）。[record.py](src/python/record.py) 程序会每隔1秒随机报出一个在战备字典中注册过的战备的呼号，只需要念出呼号程序便会自动录制。注意，0号索引是环境白噪音，当 [record.py](src/python/record.py) 随机出“白噪音“的呼号时保持沉默即可。另外14号索引是语音助手的激活指令，这里也加入训练集训练，当record.py 随机出“指令”的呼号时只需要念“指令”两个字就行。
3. 对于默认的网络，每一个战备的样本在300条以上准确率会比较高
4. 推荐开着绝地潜兵的视频录制训练集，以充分学习战场白噪音

## 训练
1. 运行 [train.py](src/python/train.py) 即可使用默认 CRNN 进行训练，默认100轮。所有神经网络有关代码都在 [nn](src/python/nn) 包下，有能力可自行修改。

## 使用
1. 运行 [main.py](src/python/main.py) 并等待 “🎙️开始监听麦克风...” 的提示出现，此时程序便开始监听。
2. 进入游戏后需要呼叫战备时首先使用“指令”激活，随即念出战备呼号（在战备字典注册的）程序便会自动激活战备。例如需要呼叫地狱火就先念“指令”，等待游戏中出现战备菜单便表示程序收到指令，正在等待战备呼号。此时再念“地狱火”便可以成功呼叫战备。
3. 程序实现原理是模拟键盘输入，因此需要将战备的上下左右输入键更换为键盘的方向键，战备呼叫键更换为ctrl键，这样便可实现一边移动一边搓球。