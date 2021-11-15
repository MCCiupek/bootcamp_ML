import sys


def add_lists(list_x, list_y):
    return [x + y for (x, y) in zip(list_x, list_y)]


def mul_lists(list_x, scalar):
    return [x * scalar for x in list_x]


def dot_prod(list_x, list_y):
    return [x * y for (x, y) in zip(list_x, list_y)]


def check_type(data, typ):
    return any(isinstance(val, typ) for val in data)


def transpose_row(list_x):
    return [[x] for x in list_x]


def transpose_col(list_x):
    return [x for x in list_x]


class Matrix:
    def __init__(self, args=None):
        if isinstance(args, list) and isinstance(args[0], list):
            self.__init_list__(args)
        elif isinstance(args, list) and isinstance(args[0], float):
            self.__init_list__([args])
        elif isinstance(args, tuple) and args > (0, 0):
            self.__init_size__(args)
        else:
            raise ValueError("Couldn't instanciate object Matrix")

    def __init_list__(self, data:list):
        self.data = data
        self.shape = (len(data), len(data[0]))
        if not self.check_type(float):
            raise ValueError("data must be only floats")

    def __init_size__(self, size:tuple):
        self.data = [ [0.0] * size[1] ] * size[0]
        self.shape = (size[0], size[1])
        if not self.check_type(float):
            raise ValueError("data must be only floats")

    def check_type(self, typ):
        return any(check_type(self.data[i], float) for i in range(0, self.shape[0]))

    def __add__(self, mat):
        if not isinstance(mat, Matrix):
            raise ValueError("Add/Sub with non Matrix object.")
            return None
        if self.shape != mat.shape:
            raise ValueError("Add/Sub with mismatching dimensions.")
            return None
        add = [add_lists(self.data[i], mat.data[i]) for i in range(0, self.shape[0])]
        if len(add) == 1 or len(add[0]) == 1:
            return Vector(add)
        else:
            return Matrix(add)

    def __radd__(self, mat):
        return mat.__add__(self)
    
    def __sub__(self, mat):
        return self + ((-1.0) * mat)
    
    def __rsub__(self, mat):
        return mat.__sub__(self)

    def get_row(self, i):
        return self.data[i]

    def get_col(self, i):
        return [row[i] for row in self.data]

    def __mul__(self, other):
        if isinstance(other, float):
            if self.shape[0] == 1 or self.shape[1] == 1:
                return Vector([mul_lists(self.data[i], float(other)) for i in range(0, self.shape[0])])
            return Matrix([mul_lists(self.data[i], float(other)) for i in range(0, self.shape[0])])
        if isinstance(other, Matrix) or isinstance(other, Vector):
            if other.shape[0] != self.shape[1]:
                sys.stderr.write("Product with mismatching dimensions.")
                return None
            prod = [ [0.0] * other.shape[1] ] * self.shape[0]
            for i in range(0, self.shape[0]):
                prod[i] = [sum(dot_prod(self.get_row(i), other.get_col(j))) for j in range(0, other.shape[1])]
        if len(prod) == 1 or len(prod[0]) == 1:
            return Vector(prod)
        return Matrix(prod)
    
    def __rmul__(self, other):
        if isinstance(other, float):
            return self * other
        return other.__mul__(self)

    def __truediv__(self, scalar):
        try:
            float(scalar)
        except ValueError:
            raise ValueError("__div__: Division by non scalar object.")
            return None
        if scalar == 0:
            raise ValueError("__div__: Division by zero.")
            return None
        if type(self) == Matrix:
            return Matrix([mul_lists(self.data[i], 1 / float(scalar)) for i in range(0, self.shape[0])])
        elif type(self) == Vector:
            return Vector([mul_lists(self.data[i], 1 / float(scalar)) for i in range(0, self.shape[0])])

    def __rtruediv__(self, scalar):
        return None

    def __str__(self):
        if type(self) == Vector:
            return "Vector{0}: {1}".format(self.shape, self.data)
        return "Matrix{0}: {1}".format(self.shape, self.data)
    
    def __repr__(self):
        if type(self) == Vector:
            return "Vector.Vector({0})".format(self.data)
        return "Matrix.Matrix({0})".format(self.data)

    def T(self):
        transp = [self.get_col(i) for i in range(0, self.shape[1])]
        if len(transp) == 1 or len(transp[0]) == 1:
            return Vector(transp)
        return Matrix(transp)


class Vector(Matrix):
    def __init__(self, data):
        try:
            if data is None:
                raise ValueError("Couldn't instanciate object Vector")
            Matrix.__init__(self, data)
            if self.shape[0] != 1 and self.shape[1] != 1:
                raise ValueError("Couldn't instanciate object Vector")
        except ValueError as err:
            sys.stderr.write("Error: {0}\n".format(err))
            sys.exit()

    def dot(self, vec):
        try:
            if not isinstance(vec, Vector):
                raise ValueError("Dot product with non vector object.")
            elif self.shape != vec.shape:
                raise ValueError("Dot product with mismatching dimensions.")
            elif self.shape[0] == 0:
                return 0
            elif self.shape[0] == 1:
                values = dot_prod(self.values, vec.values)
            else:
                values = [dot_prod(self.values[i], vec.values[i])
                               for i in range(0, self.shape[0])]
                values = [v[0] for v in values]
            return sum(values)
        except ValueError as err:
            sys.stderr.write("Error: {0}\n".format(err))
            return None
