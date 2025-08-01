def to_list(x, ndim: int) -> list[int | float] | list[list[int | float]]:
    if isinstance(x, (int, float)):
        return [x] * ndim
    elif isinstance(x, (list, tuple)):
        if len(x) != ndim:
            if len(x) > 0 and isinstance(x[0], (list, tuple)) and len(x[0]) == ndim:
                # If the first element is a list or tuple, we assume it's a nested structure
                # and we need to convert each element to a list
                assert all(
                    len(el) == ndim for el in x
                ), f"got nested structure with lengths {[len(el) for el in x]} but expected {ndim} for all elements"
                return [list(el) for el in x]
        return list(x)
    else:
        raise TypeError(
            f"expected int | float, list[int | float], or list[list[int | float]] but got {type(x)}"
        )


def to_tuple(
    x, ndim: int
) -> tuple[int | float, ...] | tuple[tuple[int | float, ...], ...]:
    if isinstance(x, (int, float)):
        return (x,) * ndim
    elif isinstance(x, (list, tuple)):
        if len(x) != ndim:
            if isinstance(x[0], (list, tuple)) and len(x[0]) == ndim:
                # If the first element is a list or tuple, we assume it's a nested structure
                # and we need to convert each element to a tuple
                assert all(
                    len(el) == ndim for el in x
                ), f"got nested structure with lengths {[len(el) for el in x]} but expected {ndim} for all elements"
                return tuple(tuple(el) for el in x)
            raise ValueError(f"got {len(x)} but expected {ndim}")
        return tuple(x)
    else:
        raise TypeError(
            f"expected int | float, list[int | float], or list[list[int | float]] but got {type(x)}"
        )
