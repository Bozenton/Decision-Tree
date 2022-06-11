# Decision-Tree
Python implementation of decision tree

```mermaid
graph LR
root--root-->a(house)
a(house)--house==0-->a0(work)
a0(work)--work==0-->a00('0')
a0(work)--work==1-->a01('1')
a(house)--house==1-->a1('1')

```

```mermaid
graph TB
root--root-->a(petal width)
a(petal width)--petal_width<0.7720-->a0(Iris-setosa)
a(petal width)--0.7720<=petal_width<1.6615-->a1(sepal length)
a1(sepal length)--sepal_length<5.8894-->a10(Iris-versicolor)
a1(sepal length)--sepal_length>=5.8894-->a11(sepal width)
a11(sepal width)--sepal_width<3.1475-->a110(petal length)
a110(petal length)--petal_length>=3.2097-->a1100(Iris-versicolor)
a11(sepal width)--sepal_width>=3.1475-->a111(Iris-versicolor)
a(petal width)--petal_width>=1.6615-->a2(sepal length)
a2(sepal length)--sepal_length<5.8894-->a20(Iris-virginica)
a2(sepal length)--sepal_length>=5.8894-->a21(sepal width)
a21(sepal width)--sepal_width<3.1475-->a210(petal length)
a210(petal length)--petal_length>=3.2097-->a2100(Iris-virginica)
a21(sepal width)--sepal_width>=3.1475-->a211(petal length)
a211(petal length)--petal_length>=3.2097-->a2110(Iris-virginica)
```

```mermaid
graph TB
root--root-->a(petal width)
a(petal width)--petal_width<0.9600-->a0(Iris-setosa)
a(petal width)--petal_width>=0.9600-->a1(sepal length)
a1(sepal length)--sepal_length<5.9173-->a10(Iris-versicolor)
a1(sepal length)--sepal_length>=5.9173-->a11(Iris-virginica)

```

