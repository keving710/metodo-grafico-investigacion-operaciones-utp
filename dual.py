"""
dual.py
=======
Conversión Primal → Dual siguiendo las reglas del documento de la profesora.
Solo construye y devuelve el PLANTEAMIENTO del primal y del dual.
No realiza iteraciones Simplex.

REGLAS (del PDF):
─────────────────
Primal MAX → Dual MIN
  · Restricción primal <=  →  Y_i >= 0,  restricción dual >=
  · Restricción primal >=  →  Y_i <= 0,  restricción dual >=
  · Restricción primal  =  →  Y_i libre, restricción dual >=

Primal MIN → Dual MAX
  · Restricción primal >=  →  Y_i >= 0,  restricción dual <=
  · Restricción primal <=  →  Y_i <= 0,  restricción dual <=
  · Restricción primal  =  →  Y_i libre, restricción dual <=

Estructura del dual:
  · # variables dual  = # restricciones primal
  · # restricciones dual = # variables primal
  · c_dual = b_primal
  · b_dual = c_primal
  · A_dual = A_primal^T
"""

import numpy as np


# ──────────────────────────────────────────────
#  UTILIDADES
# ──────────────────────────────────────────────

def _fmt(v):
    """Entero si lo es, si no hasta 4 decimales."""
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return round(float(v), 4)


# ──────────────────────────────────────────────
#  CONSTRUCCIÓN DEL DUAL
# ──────────────────────────────────────────────

def construir_dual(tipo_primal, c, A, b, tipos_rest):
    """
    Parámetros
    ----------
    tipo_primal : "max" | "min"
    c           : lista de coef. FO primal  (longitud n)
    A           : matriz de restricciones   (m × n)
    b           : RHS del primal            (longitud m)
    tipos_rest  : lista de "<=" | ">=" | "=" (longitud m)

    Retorna
    -------
    tipo_dual       : "min" | "max"
    c_dual          : coef. FO dual   (= b_primal)
    A_dual          : matriz dual     (= A^T, shape n × m)
    b_dual          : RHS dual        (= c_primal)
    tipos_rest_dual : tipos de restricciones duales
    signos_yi       : descripción del signo de cada Y_i
    """
    m = len(b)   # restricciones primal = variables dual
    n = len(c)   # variables primal     = restricciones dual

    # 1. Tipo del dual (inverso del primal)
    tipo_dual = "min" if tipo_primal == "max" else "max"

    # 2. FO dual: coeficientes = b_primal
    c_dual = [_fmt(v) for v in b]

    # 3. Matriz dual: transpuesta de A
    A_np   = np.array(A, dtype=float)
    A_dual = [[_fmt(v) for v in fila] for fila in A_np.T.tolist()]  # shape n × m

    # 4. RHS dual = c_primal
    b_dual = [_fmt(v) for v in c]

    # 5. Signos de variables Y_i y tipo de restricciones duales
    signos_yi       = []
    tipos_rest_dual = []

    if tipo_primal == "max":
        # Dual MIN → restricciones duales >=
        for t in tipos_rest:
            if   t == "<=": signos_yi.append("≥ 0")
            elif t == ">=": signos_yi.append("≤ 0")
            else:           signos_yi.append("No restringida")
        tipos_rest_dual = [">="] * n

    else:  # tipo_primal == "min"
        # Dual MAX → restricciones duales <=
        for t in tipos_rest:
            if   t == ">=": signos_yi.append("≥ 0")
            elif t == "<=": signos_yi.append("≤ 0")
            else:           signos_yi.append("No restringida")
        tipos_rest_dual = ["<="] * n

    return tipo_dual, c_dual, A_dual, b_dual, tipos_rest_dual, signos_yi


# ──────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL  (llamada desde app.py)
# ──────────────────────────────────────────────

def resolver_dual(form):
    """
    Lee el formulario Flask, construye el dual y devuelve solo
    el planteamiento (sin iteraciones Simplex).

    Retorna
    -------
    resultado   : dict con primal y dual completos para la plantilla
    []          : lista vacía de iteraciones
    []          : lista vacía de bases
    []          : lista vacía de encabezados
    """
    # ── Leer primal desde el formulario ──────────────────────
    tipo_primal    = form["tipo"]                         # "max" | "min"
    n_vars_primal  = int(form["num_variables"])
    n_rest_primal  = int(form["num_restricciones"])

    c_primal = [float(form[f"z{j}"]) for j in range(n_vars_primal)]

    A_primal     = []
    b_primal     = []
    tipos_primal = []
    for i in range(n_rest_primal):
        fila = [float(form[f"r{i}_{j}"]) for j in range(n_vars_primal)]
        A_primal.append(fila)
        b_primal.append(float(form[f"sol{i}"]))
        tipos_primal.append(form.get(f"tipo_rest{i}", "<="))

    # ── Construir el dual ────────────────────────────────────
    tipo_dual, c_dual, A_dual, b_dual, tipos_dual, signos_yi = construir_dual(
        tipo_primal, c_primal, A_primal, b_primal, tipos_primal
    )

    n_vars_dual = n_rest_primal   # Y1 … Ym
    n_rest_dual = n_vars_primal   # una restricción por variable primal

    # ── Empaquetar para la plantilla ─────────────────────────
    resultado = {
        # Info del primal (para mostrar)
        "primal": {
            "tipo":   tipo_primal,
            "c":      [_fmt(v) for v in c_primal],
            "A":      [[_fmt(v) for v in fila] for fila in A_primal],
            "b":      [_fmt(v) for v in b_primal],
            "tipos":  tipos_primal,
            "n_vars": n_vars_primal,
            "n_rest": n_rest_primal,
        },

        # Info del dual construido (para mostrar)
        "dual": {
            "tipo":   tipo_dual,
            "c":      c_dual,        # coef FO dual  (= b_primal)
            "A":      A_dual,        # matriz dual   (= A^T)
            "b":      b_dual,        # RHS dual      (= c_primal)
            "tipos":  tipos_dual,
            "n_vars": n_vars_dual,
            "n_rest": n_rest_dual,
        },

        # Signo de cada variable dual Y_i
        "signos_yi": signos_yi,
    }

    # Devuelve listas vacías en lugar de iteraciones Simplex
    return resultado, [], [], []