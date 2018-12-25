# 写一个可以安装的python package

## 一个package的结构

- `setup.py`

- `reame.md`
- package
  - 
  - `version.py`
  - other files
  - sub_package
    - `__init__.py`
    - module1
    - module2
  - package_data
    - data1

    - data2


![tree](https://raw.githubusercontent.com/BridgeMia/Web-learning-of-Mia/master/pictures/tree.PNG)

例如上图所示的package结构

## 版本

参考numpy的写法，是有一个专门的`version.py`，然后在`__init__.py`中import: 

```python
from .version import version as __version__
```



## setup.py

`setup.py`中可以写很多东西，但是比较重要的是需要指定package和package data，例如上面的package有一个sub-package，就应该这样写：

```python
from setuptools import setup

setup(name='sample_package',
      packages=['sample_package', 'sample_package.utils'],
      package_data = {'sample_package': ['package_data/*.txt']})
```



## 如何安装

在setup.py目录下

编译：

```bash
python setup.py build
```

安装：

```bash
python setup.py install
```

