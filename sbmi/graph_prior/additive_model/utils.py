from sympy import Number


def round_expr(expr, num_digits):
    """
    rounds a sympy expression to num_digits per "atom"
    """
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(Number)})
