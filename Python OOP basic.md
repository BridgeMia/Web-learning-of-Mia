*Python 的面向对象编程的一些知识*

# Python OOP Basic    

## 概述    

介绍一些简单的Python面向对象的知识点

## 私有

OOP的一个好处就是避免了类里面的某些变量或者方法被访问或者调用，从而让程序更安全，这时候就需要引入私有的概念，在C++和Java中可以
明确通过private关键词来实现私有，但是Python中没有类似的方法，只能通过下划线的方式来实现伪私有。    

- 单下划线： 
  单下划线 "_" 意味着这种属性或者方法**不应该被直接调用**，而是应该通过专门的接口调用，虽然直接调用不会报错，但是是不符合规范的。
  在访问这些私有的属性或者方法的时候，可以通过专门的接口实现，下面的例子中用到了属性（property），在后面会具体介绍。

```python
class A:

	def __init__(self,val1): 
		self._protected_value = val1
		
	@ property
	def protected_value(): 
		return self._protected_value

if __name__ == '__main__': 
	sample = A("value1")
	
	print(sample._protected_value) # This method is not advised
	print(sample.protected_value) # advised method
	
	# Out: value2
	# So the _protected_value is not protected in this way
	sample._protected_value = "value2"
	print(sample._protected_value) 
	
	# Will cause "AttributeError: can't set attribute"
	# Thus protected_value is protected
	sample.protedted_value = "value2" 
	
```

- 双下划线 __ ： 
  双下划线 "__" 可以实现真正的私有，因为**从外部是没有办法访问带双下划线的变量或者方法**的（作为对比，单下划线的是可以的），但是网上的例子说双下划线
   的主要目的是为了**保护类中的变量或者方法在继承的时候不会被覆盖**，可以直接看例子：

```python
class A:
	def __init__(self):
		pass
	def __method(self):
		print("This is the method of class A")
		
	def method(self):
		self.__method()

class B(A):
	def __init__(self):
		# Initialize parent class first
		A.__init__(self)
		pass
		
	def __method(self):
		print("This is the method of class B inherited from A")
	

if __name__ == '__main__':
	a = A()
	b = B()
	
	# Out: This is the method of class A
	a.method()
	# Will cause AttributeError: 'A' object has no attribute '__method'
	a.__method()
	
	# Out: This is the method of class A
	# We can see the __method cannot be rewrite when inherited
	b.method()
	# Will cause AttributeError: 'B' object has no attribute '__method'
	b.__method()
```

 上面的例子是一个带双下划线的类私有方法的例子，为了实现类的私有变量，也可以用双下划线实现，在访问时另外写接口

- 前后都有双下划线的方法或变量    
  这种方法或者变量是**Python调用的，不是用户调用的**，我们常用的两种是__init__和__call__
  ​     - 初始化`__init__` : 这个不做过多的介绍
  ​     - 调用`__call__` : 这个函数可以像调用函数一样调用一个类，可以参考下面的代码，这种写法在做模型的时候比较方便
  - 其他的这类方法还可以是 `__add__` `__sub__` 等，在用的时候，就直接用+和-就能实现你自己定义的加法和减法（比如模型的加和）

```python
class a_b_c:
	def __init__(self, a, b, c):
		self.a = a
		self.b = b
		self.c = c
		
	def __call__(self, alpha):
		return alpha * self.a * self.b * self.c

if __name__ == '__main__':
	ret = a_b_c(2, 3, 4)
	
	# Out: 12
	print(ret(0.5)
```

## 属性（property)

属性也是OOP中比较方便的一种写法，个人的理解是把类中的函数当做一个变量来调用，除了上面的例子中为私有的变量或者函数提供接口之外，
还可以用来方便的返回一些计算**固定**的值。注意因为他还是一个函数，所以并不能修改，就比较安全。

- 为私有方法/变量提供访问接口    

```python
class A: 
	def __init__(self, private_val):
		self.__private_val = private_val
		
	# Use decorator
	@property
	def private_val(self)
		return self.__private_val
		
class B:
	def __init__(self, a, b):
		self.a = a
		self.b = b
	
	def __mul(self):
		return self.a * self.b
	
	prod = property(__mul)

if __name__ == '__main__': 
	a = A("value1")
	
	# Out: value1
	print(a.private_val)
	
	b = B(4, 5)
	
	# Out: 20
	b.prod
```

- 返回一个计算好的值

```python
class square:
	def __init__(self, a, b):
		self.a = a
		self.b = b
	
	@property
	def area(self)
		return self.a * self.b

if __name__ == '__main__':
	square_1 = square(4, 5)
	print(square_1.area)
```

## 类中的三种方法（函数）

### 成员方法

是实例的方法，就是说实例化之后才能调用的方法，就是我们定义一个类中最普通的方法。    

- **关于self**   
  **这个有点纠结，觉得复杂可以直接看结论。我们可以统一约定一下，成员函数的第一个参数就是self**   

  下面开始比较纠结的部分。一般来说感觉上成员函数好像总是跟self这个参数关联起来的，但是实际上self只是一个参数，让你在这个函数内部能访问到这个函数外部，又在类的内部的一些变量或者方法，跟后面可能出现的其他参数是没有本质区别的，跟这个函数是不是成员函数也没有关系。下面看两个比较纠结的例子：

```python
class Box:
	def __init__(self, size, color):
		self.size = size
		self.color = color
		
	def normal_open(self): 
		print('box %s - %s opened'%(self.size, self.color))
	
	def self_open(self, box_self):
		print('box %s - %s opened'%(box_self.size, box_self.color))

if __name__ == '__main__':
	sample_box = Box(12, 'red')
    
    # Out: box 12 - red opened
	sample_box.normal_open()
    
    # Out: box 12 - red opened
	sample_box.self_open(sample_box)
    
    # Out: {'size': 12, 'color': red}
	print(sample_box.__dict__)
```

上面的看起来很别扭，而且`self_open`这个函数其实是不太好的，但是最后打印的dict其实可看做self的内容。所以self可以看做是一个包含了我们定义的这个类在实例化之后的对象的所有属性。可以看接下来的这个例子：

```python
class Box:
    
  	def __init__(self):
        pass
    
	def box_init(box_self, size, color):
		box_self.size = size
        box_self.color = color
        return box_self
    
    def self_open(self, box_self):
        print(self)
        
        print('box %s - %s opened'%(self.size, self.color))
        print('box %s - %s opened'%(box_self.size, box_self.color))

if __name__ == '__main__':
    sample_box = Box()

    # Will cause AttibuteError: 'Box' object has not attribute 'size'
    sample_box.self_open(sample_box)

    sample_box.box_init(12, 'red')
    
    # Out: 
    # <__main__.Box object at xxxxxxxxx>
    # box 12 - red opened
    sample_box.self_open(sample_box)
    
    
    # Out: 
    # <__main__.Box object at xxxxxxxxx>
    # box 12 - red opened
    #sample_box_1.self_open(sample_box_1)
```

这个例子里面，正常实例化的时候，Box类的object是没有size和color这两个属性的，但是在你调用自己定义的`box_init`的时候，（注意我们的第一个参数是`box_self`而不是`self`），其实是把你之前实例化之后的object（也就是`sample_object`）传进了这个函数，让这个object具有了color和size这两个属性，在`self_open`函数中，同样传入了的是这个object，这个时候就可以正常打印size和color了。

**结论**：我们在类中**定义成员函数**的时候，第一个参数不管你叫他self也好，叫他my_self也好，都是这个函数在实例化之后的对象。在实例化之后，在**调用这个对象的成员函数**的时候，这个对象会作为self（或者是你定义的任何名字）被传入这个函数中，这个过程是自动的，不需要你手动加上`self = object_name`。

### 静态方法

静态方法（static method）是类中不需要实例化就可以调用的函数，具体可以直接看下面的例子：

```python
from datetime import date,datetime

class my_date:

    def __init__(self, year, month, day):
        self.day = day
        self.month = month
        self.year = year

    def str_format_print(self, split = '-'):
        print(self.year,split,self.month,split,self.day)

    @staticmethod
    def get_current_date():
        print(datetime.now())

if __name__ == '__main__':
    # Out: Current date and time
    my_date.get_current_date()

    my_date_sample = my_date(2018,11,21)
    
    # Out: 2018 - 11 - 21
    my_date_sample.str_format_print()
```

也就是有时候我们定义的类中的函数可能在调用的时候并不需要实例化，这时候就可以把它定义成静态方法。



### 类方法

类方法（class method）是用来给类自己的方法，传入的参数是一个class（一般写成cls），可以看下面的代码：

```python
from datetime import date,datetime

class my_date:

    def __init__(self, year, month, day):
        self.day = day
        self.month = month
        self.year = year

    def str_format_print(self, split = '-'):
        print(self.year,split,self.month,split,self.day)

    @staticmethod
    def get_current_date():
        print(datetime.now())

    @classmethod
    def init_with_str_date(cls, str_date):
        year, month, day = [int(x) for x in str_date.split('-')]
        date1 = cls(year, month, day)
        return date1

if __name__ == '__main__':
    # Out: Current date
    my_date.get_current_date()

    my_date_sample = my_date(2018,11,21)
    
    # Out: 2018 - 11 - 21
    my_date_sample.str_format_print()

    
    my_date_sample1 = my_date.init_with_str_date('2018-11-21')

    # Out: 2018 11 21
    print(my_date_sample1.year, my_date_sample1.month, my_date_sample1.day)
    # Out: 2018 - 11 - 21
    my_date_sample.str_format_print()
    
```

这里我们定义了一个新的date类，可以从一个字符串来初始化，这样的好处在于，我们不用对类本身的成员方法做出修改，在重构这个类的时候，要添加新的方法可以直接加而不影响类实例化之后的对象。



### TODO

后面会接着介绍父类子类和继承之类的知识

