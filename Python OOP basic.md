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
     - 初始化`__init__` : 这个不做过多的介绍
     - 调用`__call__` : 这个函数可以像调用函数一样调用一个类，可以参考下面的代码，这种写法在做模型的时候比较方便
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
还可以用来方便的返回一些比较计算方法固定的值。
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

## 静态方法（staticmethod）
