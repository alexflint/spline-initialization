import numpy as np


def skew(m):
    # Compute the skew-symmetric matrix for m
    m = np.asarray(m)
    assert m.shape == (3,), 'shape was was %s' % str(m.shape)
    return np.array([[ 0,    -m[2],  m[1]],
                     [ m[2],  0,    -m[0]],
                     [-m[1],  m[0],  0.  ]])


class LieGroup(object):
    # Abstract base class for lie groups
    # Returns a function f: m -> R0 * exp(m)
    @classmethod
    def right_chart(cls, *args):
        if len(args) == 0:
            X0 = cls.identity()
        else:
            X0 = cls.unpack(*args)
        f = lambda x: cls.dot(X0, cls.exp(x))
        f.dimensions = cls.dimensions()
        return f

    # Returns a function f: m -> exp(m) * R0
    @classmethod
    def left_chart(cls, *args):
        if len(args) == 0:
            X0 = cls.identity()
        else:
            X0 = cls.unpack(*args)
        f = lambda x: cls.dot(cls.exp(x), X0)
        f.dimensions = cls.dimensions()
        return f

    # Returns a function f: m -> exp(m) * R0
    @classmethod
    def left_neg_chart(cls, *args):
        if len(args) == 0:
            X0 = cls.identity()
        else:
            X0 = cls.unpack(*args)
        f = lambda x: cls.dot(cls.exp(-x), X0)
        f.dimensions = cls.dimensions()
        return f

    # Returns a function f: m -> X0 * exp(m)
    @classmethod
    def chart(cls, X0):
        return cls.right_chart(X0)

    def inv(self):
        return self.__class__.inverse(self)

    def __mul__(self, x):
        try:
            return self.__class__.dot(self, x)
        except:
            return NotImplemented

    def __rmul__(self, x):
        try:
            return self.__class__.dot(x, self)
        except:
            return NotImplemented

    def __repr__(self):
        return repr(self.mat).replace('array', self.__class__.__name__)

    def __str__(self):
        return str(self.mat)



class SO3(LieGroup):
    generators = np.array([[[ 0.,  0.,  0. ],
                            [ 0.,  0., -1. ],
                            [ 0.,  1.,  0. ]],
                           
                           [[ 0.,  0.,  1. ],
                            [ 0.,  0.,  0. ],
                            [ -1., 0.,  0. ]],
                           
                           [[ 0., -1.,  0. ],
                            [ 1.,  0.,  0. ],
                            [ 0.,  0.,  0. ]]])

    def __init__(self, *args):
        '''Construct an SO3 element from a rotation matrix.'''
        # TODO: permit quaternions as input to this function
        if len(args) == 0:
            self._R = np.eye(3)
        else:
            self._R = SO3.unpack(*args)

    @property
    def mat(self):
        return self._R

    @property
    def T(self):
        return self.inv()

    # Manifold dimensionality
    @classmethod
    def dimensions(cls):
        return 3

    # Group identity
    @classmethod
    def identity(cls):
        return cls.pack(np.eye(3))

    # Group multiplication
    @classmethod
    def dot(cls, X, Y):
        return np.dot(cls.unpack(X), cls.unpack(Y))

    # Group inverse element
    @classmethod
    def inverse(cls, X):
        return cls.unpack(X).T

    # Compute the mapping from so(3) to SO(3)
    @classmethod
    def exp(cls, m):
        m = np.asarray(m)
        assert np.shape(m) == (3,), 'SO3.exp(m) called with m='+str(np.shape(m))

        tsq = np.dot(m,m)
        if tsq < 1e-8:
            # Taylor expansion of sin(sqrt(x))/sqrt(x):
            #   http://www.wolframalpha.com/input/?i=sin(sqrt(x))/sqrt(x)
            a = 1. - tsq/6. + tsq*tsq/120.;
        
            # Taylor expansion of (1 - cos(sqrt(x))/x:
            #   http://www.wolframalpha.com/input/?i=(1-cos(sqrt(x)))/x
            b = .5 - tsq/24. + tsq*tsq/720.;
        else:
            t = np.sqrt(tsq)
            a = np.sin(t)/t
            b = (1. - np.cos(t)) / tsq

        M = skew(m)
        return cls.pack(np.eye(3) + a*M + b*np.dot(M,M))

    # Compute the mapping from SO(3) to so(3)
    @classmethod
    def log(cls, R):
        R = cls.unpack(R)

        # http://math.stackexchange.com/questions/83874/
        t = R.trace()
        r = np.array(( R[2,1] - R[1,2],
                       R[0,2] - R[2,0],
                       R[1,0] - R[0,1] ))
        if t >= 3. - 1e-8:
            return (.5 - (t-3.)/12.) * r
        elif t > -1. + 1e-8:
            th = np.arccos(t/2. - .5)
            return th / (2. * np.sin(th)) * r
        else:
            assert t <= -1. + 1e-8
            a = np.argmax(R[ np.diag_indices_from(R) ])
            b = (a+1) % 3
            c = (a+2) % 3
            s = np.sqrt(R[a,a] - R[b,b] - R[c,c] + 1.)
            v = np.empty(3)
            v[a] = s/2.
            v[b] = (R[b,a] + R[a,b]) / (2.*s)
            v[c] = (R[c,a] + R[a,c]) / (2.*s)
            return v / np.linalg.norm(v)

    # Compute jacobian of exp(m)*x with respect to m, evaluated at
    # m=[0,0,0]. x is assumed constant with respect to m.
    @classmethod
    def J_expm_x(cls, x):
        return skew(-x)

    # Return the generators times x
    @classmethod
    def generator_field(cls, x):
        return skew(x)
        
    @classmethod
    def pack(cls, X):
        assert isinstance(X, np.ndarray) and X.shape == (3,3)
        return X

    @classmethod
    def unpack(cls, X):
        assert isinstance(X, np.ndarray) or isinstance(X, SO3)
        if isinstance(X, SO3):
            return X.mat
        else:
            assert X.shape == (3,3)
            return X


class SE3(LieGroup):
    def __init__(self, *args):
        if len(args) == 0:
            self._R = SO3.eye()
            self._t = np.zeros(3)
        else:
            self._R, self._t = SE3.unpack(*args)
        
    @property
    def mat(self):
        return np.hstack((self.R, self.t[:,np.newaxis]))

    @property
    def R(self):
        return self._R

    @property
    def t(self):
        return self._t

    @property
    def Rt(self):
        return (self.R, self.t)

    # Manifold dimensionality
    @classmethod
    def dimensions(cls):
        return 6

    # Compute the identity
    @classmethod
    def identity(cls):
        return cls.pack(SO3.identity(), np.zeros(3))

    # Compute the group multiplication operation
    @classmethod
    def dot(cls, X, Y):
        RX,tx = cls.unpack(X)
        RY,ty = cls.unpack(Y)
        return cls.pack(SO3.dot(RX,RY), np.dot(RX,ty) + tx)

    @classmethod
    def inverse(cls, X):
        R,t = cls.unpack(X)
        return cls.pack(SO3.inverse(R), -np.dot(SO3.inverse(R), t))

    # Mapping from se(3) to SE(3)
    @classmethod
    def exp(cls, x):
        x = np.asarray(x)
        assert np.shape(x) == (6,), 'shape was '+str(x.shape)
        return cls.pack(SO3.exp(x[:3]), x[3:])

    # Mapping from SE(3) to se(3)
    @classmethod
    def log(cls, X):
        R,t = cls.unpack(X)
        return np.hstack((SO3.log(R), t))

    @classmethod
    def pack(cls, R, t):
        return SE3(R,t)

    @classmethod
    def unpack(cls, *args):
        assert len(args) in (1,2)
        if len(args) == 2:
            assert np.shape(args[1]) == (3,), 'args='+str(args)
            return SO3.unpack(args[0]), np.asarray(args[1])
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, SE3):
                return arg.Rt
            elif isinstance(arg, SO3):
                return arg.matrix, np.zeros(3)
            if isinstance(arg, tuple):
                assert len(arg) == 2 and np.shape(arg[1]) == (3,)
                return SO3.unpack(arg[0]), np.asarray(arg[1])
            elif isinstance(arg, np.ndarray):
                # We have a matrix. Permitted shapes are 3x3, 3x4, 4x4
                assert arg.shape == (3,3) or arg.shape == (3,4) or arg.shape == (4,4)
                if arg.shape == (3,3):
                    return arg, np.zeros(3)
                if arg.shape == (3,4):
                    return arg[:,:3], arg[:,3]
                elif arg.shape == (4,4):
                    # last row must be (0 0 0 1)
                    assert np.linalg.norm(arg[3] - (0,0,0,1)) < 1e-8
                    return arg[:3,:3], arg[:3,3]
