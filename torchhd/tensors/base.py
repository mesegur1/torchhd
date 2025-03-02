#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from typing import List, Set, Any, Sequence
import torch
from torch import Tensor
from torch.utils._pytree import register_pytree_node

class VSATensor(object):
    """Base class

    Each model must implement the methods specified on this base class.
    """

    supported_dtypes: Set[torch.dtype]

    def __init__(self, tensor):
        self.tensor = tensor

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.tensor)})"
    
    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def device(self):
        return self.tensor.device
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        # Check if any of the types are not torch.Tensor or VSATensor
        if not all(issubclass(t, (torch.Tensor, VSATensor)) for t in types):
            return NotImplemented
        
        # Unwrap VSATensor instances to their underlying tensors
        def unwrap(e):
            if isinstance(e, VSATensor):
                return e.tensor
            elif isinstance(e, Sequence):
                return type(e)(unwrap(x) for x in e)
            else:
                return e
        
        unwrapped_args = tuple(unwrap(arg) for arg in args)
        unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        
        # Call the original function
        result = func(*unwrapped_args, **unwrapped_kwargs)
        
        # Wrap the result back into VSATensor if it's a tensor
        def wrap(e):
            if isinstance(e, torch.Tensor):
                return cls(e)
            elif isinstance(e, Sequence):
                return type(e)(wrap(x) for x in e)
            else:
                return e
        
        return wrap(result)

    
    @classmethod
    def empty(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
    ) -> "VSATensor":
        """Creates hypervectors representing empty sets"""
        raise NotImplementedError

    @classmethod
    def identity(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
    ) -> "VSATensor":
        """Creates identity hypervectors for binding"""
        raise NotImplementedError

    @classmethod
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        generator=None,
    ) -> "VSATensor":
        """Creates random or uncorrelated hypervectors"""
        raise NotImplementedError

    def bundle(self, other: "VSATensor") -> "VSATensor":
        """Bundle the hypervector with other"""
        raise NotImplementedError

    def multibundle(self) -> "VSATensor":
        """Bundle multiple hypervectors"""
        if self.tensor.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} needs to have at least two dimensions for multibundle, got size: {tuple(self.tensor.shape)}"
            )

        n = self.tensor.size(-2)
        if n == 1:
            return self.tensor.unsqueeze(-2)

        tensors: List[VSATensor] = torch.unbind(self.tensor, dim=-2)

        output = tensors[0].bundle(tensors[1])
        for i in range(2, n):
            output = output.bundle(tensors[i])

        return output


    def bind(self, other: "VSATensor") -> "VSATensor":
        """Bind the hypervector with other"""
        raise NotImplementedError

    def multibind(self) -> "VSATensor":
        """Bind multiple hypervectors"""
        if self.tensor.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.tensor.shape)}"
            )

        n = self.tensor.size(-2)
        if n == 1:
            return self.tensor.unsqueeze(-2)

        tensors: List[VSATensor] = torch.unbind(self.tensor, dim=-2)

        output = tensors[0].bind(tensors[1])
        for i in range(2, n):
            output = output.bind(tensors[i])

        return output
    
    def inverse(self) -> "VSATensor":
        """Inverse the hypervector for binding"""
        raise NotImplementedError

    def negative(self) -> "VSATensor":
        """Negate the hypervector for the bundling inverse"""
        raise NotImplementedError

    def permute(self, shifts: int = 1) -> "VSATensor":
        """Permute the hypervector"""
        raise NotImplementedError

    def dot_similarity(self, others: "VSATensor") -> Tensor:
        """Inner product with other hypervectors"""
        raise NotImplementedError

    def cosine_similarity(self, others: "VSATensor") -> Tensor:
        """Cosine similarity with other hypervectors"""
        raise NotImplementedError

    # Proxy pattern for Tensor methods
    # List of methods to proxy
    _tensor_methods = [
        'abs', 'acos', 'asin', 'atan', 'ceil', 'conj', 'cos', 'cosh', 'deg2rad',
        'digamma', 'dim', 'exp', 'expm1', 'floor', 'frac', 'imag', 'log', 'log10',
        'neg', 'real', 'reciprocal', 'relu', 'round', 'sigmoid', 'sin', 'sinh',
        'sqrt', 'tan', 'tanh', 'to', 'unsqueeze', 'squeeze', 'view', 'reshape',
        'permute', 'transpose', 'expand', 'contiguous', 'clone', 'detach',
        'cpu', 'cuda', 'type', 'double', 'float', 'half', 'long', 'int', 'short',
        'char', 'byte', 'bool', 'bfloat16', 'C', 'contiguous', 'H', 'item',
        'matmul', 'mean', 'mul', 'pow', 'rsqrt', 'square', 'std', 'sum',
        # Add more methods as needed
    ]
    # Dynamically define methods
    for method_name in _tensor_methods:
        def method_factory(method_name):
            def method(self, *args, **kwargs):
                result = getattr(self.tensor, method_name)(*args, **kwargs)
                if isinstance(result, torch.Tensor):
                    return self.__class__(result)
                return result
            return method
        method = method_factory(method_name)
        # Use locals() to add the method to the class namespace
        locals()[method_name] = method

    # Arithmetic Magic Methods
    def __add__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor + other.tensor
            return self.__class__(result)
        else:
            result = self.tensor + other
            return self.__class__(result)

    def __radd__(self, other):
        # Handles cases like scalar + VSATensor
        result = other + self.tensor
        return self.__class__(result)

    
    def __sub__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor - other.tensor
            return self.__class__(result)
        else:
            result = self.tensor - other
            return self.__class__(result)

    
    def __rsub__(self, other):
        # Handles cases like scalar - VSATensor
        result = other - self.tensor
        return self.__class__(result)

    
    def __mul__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor * other.tensor
            return self.__class__(result)
        else:
            result = self.tensor * other
            return self.__class__(result)

    
    def __rmul__(self, other):
        # Handles cases like scalar * VSATensor
        result = other * self.tensor
        return self.__class__(result)

    
    def __truediv__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor / other.tensor
            return self.__class__(result)
        else:
            result = self.tensor / other
            return self.__class__(result)

    
    def __rtruediv__(self, other):
        # Handles cases like scalar / VSATensor
        result = other / self.tensor
        return self.__class__(result)

    
    def __floordiv__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor // other.tensor
            return self.__class__(result)
        else:
            result = self.tensor // other
            return self.__class__(result)

    
    def __rfloordiv__(self, other):
        result = other // self.tensor
        return self.__class__(result)

    
    def __mod__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor % other.tensor
            return self.__class__(result)
        else:
            result = self.tensor % other
            return self.__class__(result)

    
    def __rmod__(self, other):
        result = other % self.tensor
        return self.__class__(result)

    
    def __pow__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor ** other.tensor
            return self.__class__(result)
        else:
            result = self.tensor ** other
            return self.__class__(result)

    
    def __rpow__(self, other):
        result = other ** self.tensor
        return self.__class__(result)

    # Unary Operations
    
    def __neg__(self):
        result = -self.tensor
        return self.__class__(result)

    
    def __pos__(self):
        result = +self.tensor
        return self.__class__(result)

    
    def __abs__(self):
        result = abs(self.tensor)
        return self.__class__(result)

    # Comparison Magic Methods
    
    def __eq__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor == other.tensor
        else:
            result = self.tensor == other
        return result

    
    def __ne__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor != other.tensor
        else:
            result = self.tensor != other
        return result

    
    def __lt__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor < other.tensor
        else:
            result = self.tensor < other
        return result

    
    def __le__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor <= other.tensor
        else:
            result = self.tensor <= other
        return result

    
    def __gt__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor > other.tensor
        else:
            result = self.tensor > other
        return result

    
    def __ge__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor >= other.tensor
        else:
            result = self.tensor >= other
        return result

    # Matrix Multiplication
    
    def __matmul__(self, other):
        if isinstance(other, VSATensor):
            result = self.tensor @ other.tensor
            return self.__class__(result)
        else:
            result = self.tensor @ other
            return self.__class__(result)

    
    def __rmatmul__(self, other):
        result = other @ self.tensor
        return self.__class__(result)

    # In-place Operations
    
    def __iadd__(self, other):
        if isinstance(other, VSATensor):
            self.tensor += other.tensor
        else:
            self.tensor += other
        return self

    
    def __isub__(self, other):
        if isinstance(other, VSATensor):
            self.tensor -= other.tensor
        else:
            self.tensor -= other
        return self

    
    def __imul__(self, other):
        if isinstance(other, VSATensor):
            self.tensor *= other.tensor
        else:
            self.tensor *= other
        return self

    
    def __itruediv__(self, other):
        if isinstance(other, VSATensor):
            self.tensor /= other.tensor
        else:
            self.tensor /= other
        return self

    
    def __ifloordiv__(self, other):
        if isinstance(other, VSATensor):
            self.tensor //= other.tensor
        else:
            self.tensor //= other
        return self

    
    def __imod__(self, other):
        if isinstance(other, VSATensor):
            self.tensor %= other.tensor
        else:
            self.tensor %= other
        return self

    
    def __ipow__(self, other):
        if isinstance(other, VSATensor):
            self.tensor **= other.tensor
        else:
            self.tensor **= other
        return self

    
    def __imatmul__(self, other):
        if isinstance(other, VSATensor):
            self.tensor @= other.tensor
        else:
            self.tensor @= other
        return self
    
    # Misc Magic Methods:

    def __len__(self):
        return len(self.tensor)
    
    def __getitem__(self, index):
        result = self.tensor[index]
        return self.__class__(result)
    
    def __setitem__(self, index, value):
        if isinstance(value, VSATensor):
            self.tensor[index] = value.tensor
        else:
            self.tensor[index] = value

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item):
        if isinstance(item, VSATensor):
            return item.tensor in self.tensor
        else:
            return item in self.tensor

    def __bool__(self):
        return self.tensor.bool().item()

    def __int__(self):
        return int(self.tensor)

    def __float__(self):
        return float(self.tensor)


#Register for PyTree (used in TorchDynamo)   
def vsatensor_flatten(vsa_tensor):
    # Return tensors and context needed to reconstruct the instance
    # We use the class of the instance (type) as context to handle subclasses
    return (vsa_tensor.tensor,), vsa_tensor.__class__
def vsatensor_unflatten(flattened_tensors, cls):
    # Reconstruct the instance using the class and the tensor data
    return cls(flattened_tensors[0])
register_pytree_node(VSATensor, vsatensor_flatten, vsatensor_unflatten)