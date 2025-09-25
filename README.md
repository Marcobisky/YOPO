# YOPO -- You Only Pick Once

> This is the source code for the first project of DIP (Digital Image Processing) course in UESTC, UofG.

## Environment Setup

On M2-chip MacOS or Ubuntu 22.04, create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate yopo
```

## Run

```bash
python main.py
```

## Some Notes

### Code Structure

### 变量规范:

- 类名: 开头大写驼峰
- 类中 utility 性质的方法 (函数): 下划线开头
- 标量、向量、一般函数: 开头小写下划线分割
- 常量: 全大写下划线
- 类实例变量: 每个单词开头大写下划线