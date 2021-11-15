from matrix import Matrix, Vector
import sys

list1 = [0.0, 1.0, 2.0, 3.0]
list2 = [[0.0], [1.0], [2.0], [3.0]]
list3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
list4 = ["a", "b", "c"]

m1 = Matrix(list1)
m2 = Matrix(list2)
m3 = Matrix(list3)
m4 = Matrix((3, 3))
# vec = Vector(list3)

v1 = Vector(list1)
v2 = Vector(list2)

Matrices = [m1, m2, m3, m4]
Vectors = [v1, v2]

print("--- Matrices: ---")
for v in Matrices:
    print(v)

print()

print("--- Vectors: ---")
for v in Vectors:
    print(v)

print()

print("--- Additions: ---")
for v in Matrices:
    res = v + v
    print(res)

for v in Vectors:
    res = v + v
    print(res)

print()

print("--- Substractions: ---")
for v in Matrices:
    res = v - v / 2
    print(res)

for v in Vectors:
    res = v - v / 2
    print(res)

print()

print("--- Multiplications: ---")
print("---      with scalars: ---")
for v in Matrices:
    res = v * 1.5
    print(res)

for v in Matrices:
    res = 1.5 * v
    print(res)

for v in Vectors:
    res = v * 1.5
    print(res)

for v in Vectors:
    res = 1.5 * v
    print(res)

print("---      with matrices/vectors: ---")
for v in Matrices:
    res = v * v.T()
    print(res)
    #print(v, " * ", v.T(), " = ", v * v.T() )

for v in Matrices:
    res = v.T() * v
    #print(v.T(), " * ", v, " = ", v.T() * v)
    print(res)

for v in Vectors:
    res = v * v.T()
    print(res)
    #print(v, " * ", v.T(), " = ", v * v.T() )

for v in Vectors:
    res = v.T() * v
    #print(v.T(), " * ", v, " = ", v.T() * v)
    print(res)


print()

print("--- Divisions: ---")
for v in Matrices:
    res = v / 2.0
    print(res)

for v in Vectors:
    res = v / 2.0
    print(res)

print()

print("--- Transpose: ---")
for v in Matrices:
    w = v
    print(w.T())

for v in Vectors:
    w = v
    print(w.T())
