"""Collection of the core mathematical operators used throughout the code base."""

import math


from typing import Callable, Iterable, Any, List


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: Product of x and y.
    """
    return x * y


def id(x: Any) -> Any:
    """Return the input unchanged.

    Args:
        x (Any): Input value.

    Returns:
        Any: The same value as input.
    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: Sum of x and y.
    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
        x (float): Input number.

    Returns:
        float: The negation of x.
    """
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Check if x is less than y.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: 1 if x < y, otherwise 0.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if two numbers are equal.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: 1 if x == y, otherwise 0.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: Larger of x and y.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are approximately equal.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: True if |x - y| < 1e-2, otherwise False.
    """
    return math.fabs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid activation function.

    Uses a numerically stable formulation.

    Args:
        x (float): Input value.

    Returns:
        float: Sigmoid of x.
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLU activation function.

    Args:
        x (float): Input value.

    Returns:
        float: max(0, x).
    """
    return max(0.0, x)


def log(x: float) -> float:
    """Compute natural logarithm.

    Args:
        x (float): Input value.

    Returns:
        float: log(x).
    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute exponential function.

    Args:
        x (float): Input value.

    Returns:
        float: e^x.
    """
    return math.exp(x)


def inv(x: float) -> float:
    """Compute multiplicative inverse.

    Args:
        x (float): Input value.

    Returns:
        float: 1 / x.
    """
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Backward pass for log function.

    Args:
        x (float): Input value.
        y (float): Gradient from next layer.

    Returns:
        float: Gradient wrt log(x).
    """
    return (1.0 / x) * y


def inv_back(x: float, y: float) -> float:
    """Backward pass for inverse function.

    Args:
        x (float): Input value.
        y (float): Gradient from next layer.

    Returns:
        float: Gradient wrt 1/x.
    """
    return (-1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Backward pass for ReLU function.

    Args:
        x (float): Input value.
        y (float): Gradient from next layer.

    Returns:
        float: Gradient wrt ReLU(x).
    """
    return 0.0 if x <= 0 else y


def map(func: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Creates a function that applies a given function to each element of an iterable.

    Args:
        func (Callable[[float], float]): A function that takes a float and returns a float.

    Returns:
        Callable[[Iterable[float]], Iterable[float]]: A function that takes an iterable of floats
        and returns a new iterable with the function applied to each element.
    """
    def apply(ls: Iterable[float]):
        return [func(x) for x in ls]
    return apply


def zipWith(func: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Creates a function that combines elements from two iterables using a given function.

    Args:
        func (Callable[[float, float], float]): A function that takes two floats and returns a float.

    Returns:
        Callable[[Iterable[float], Iterable[float]], Iterable[float]]: A function that takes two iterables
        of floats and returns a new iterable with the function applied element-wise.
    """
    def apply(ls1: Iterable[float], ls2: Iterable[float]):
        return [func(x, y) for x, y in zip(ls1, ls2)]
    return apply


def reduce(func: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:
    """Creates a function that reduces an iterable to a single value using a given function.

    Args:
        func (Callable[[float, float], float]): A function that takes two floats and returns a float.

    Returns:
        Callable[[Iterable[float]], float]: A function that takes an iterable of floats and
        reduces it to a single float by applying the function cumulatively.
        Returns 0 if the iterable is empty.
    """
    def apply(ls: Iterable[float]):
        try:
            iterator = iter(ls)
            res = next(iterator)
        except:
            return 0.0
        for x in iterator:
            res = func(res, x)
        return res
    return apply



def negList(x: List[float]) -> List[float]:
    """Negate each element in a list.

    Args:
        x (List[float]): Input list.

    Returns:
        List[float]: Negated list.
    """
    return map(neg)(x)


def addLists(x: List[float], y: List[float]) -> List[float]:
    """Add two lists elementwise.

    Args:
        x (List[float]): First list.
        y (List[float]): Second list.

    Returns:
        List[float]: Elementwise sum.
    """
    return zipWith(add)(x, y)


def sum(x: List[float]) -> float:
    """Compute the sum of a list.

    Args:
        x (List[float]): Input list.

    Returns:
        float: Sum of elements.
    """
    return reduce(add)(x)


def prod(x: List[float]) -> float:
    """Compute the product of a list.

    Args:
        x (List[float]): Input list.

    Returns:
        float: Product of elements.
    """
    return reduce(mul)(x)
