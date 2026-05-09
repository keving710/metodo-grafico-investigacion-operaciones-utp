import numpy as np


# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────

def _fmt(v):
    """Formatea un número: entero si no tiene decimales, si no 4 dec."""
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return round(float(v), 4)


def _tabla_a_lista(tabla):
    return [[_fmt(v) for v in fila] for fila in tabla]


# ─────────────────────────────────────────────
#  CONSTRUCCIÓN DE LA TABLA FASE I
# ─────────────────────────────────────────────

def _construir_fase1(restricciones, tipos_rest):
    """
    Variables: X1..Xn | s1..sm (exceso, ≥→-1, ≤→+1) | h1..hk (holgura ≤) | A1..Ap (artificiales) | Sol
    Devuelve:
        tabla  (numpy array)
        base   (lista de índices de columna de la variable básica por fila)
        encabezados_completos
        idx_artificiales  (lista de columnas que son artificiales)
        idx_s             (lista de columnas de variables de exceso/holgura)
        n_vars, n_rest
    """
    n_rest = len(restricciones)
    n_vars = len(restricciones[0]) - 1   # sin el término independiente

    # Contamos cuántas variables artificiales necesitamos (= y >=)
    art_needed = [i for i, t in enumerate(tipos_rest) if t in (">=", "=")]
    hol_needed = [i for i, t in enumerate(tipos_rest) if t == "<="]

    n_art = len(art_needed)

    # Índices de columnas en la tabla ampliada:
    # 0..n_vars-1          → variables originales X
    # n_vars..n_vars+n_rest-1   → holguras / excesos  s_i
    # n_vars+n_rest..n_vars+n_rest+n_art-1  → artificiales A_i
    # última columna       → Sol

    total_cols = n_vars + n_rest + n_art + 1

    tabla = np.zeros((n_rest + 1, total_cols))

    art_col = {}   # fila → columna de su artificial
    art_counter = 0

    base = []

    for i, (row, tipo) in enumerate(zip(restricciones, tipos_rest)):
        # Coeficientes de variables originales
        for j in range(n_vars):
            tabla[i, j] = row[j]

        # Variable de holgura / exceso
        if tipo == "<=":
            tabla[i, n_vars + i] = 1      # holgura positiva → base inicial
            base.append(n_vars + i)
        elif tipo == ">=":
            tabla[i, n_vars + i] = -1     # exceso negativo
            # Artificial
            col_a = n_vars + n_rest + art_counter
            tabla[i, col_a] = 1
            art_col[i] = col_a
            art_counter += 1
            base.append(col_a)
        elif tipo == "=":
            # Solo artificial
            col_a = n_vars + n_rest + art_counter
            tabla[i, col_a] = 1
            art_col[i] = col_a
            art_counter += 1
            base.append(col_a)

        # Término independiente
        tabla[i, -1] = row[-1]

    # Fila Z Fase I: Min Z = sum(Artificiales)  →  Z - A1 - A2 ... = 0
    # Coeficientes en fila Z para artificiales = -1 antes de ajustar
    for col_a in art_col.values():
        tabla[-1, col_a] = -1

    # Ajustar fila Z sumando las filas de las artificiales (como hace la profe)
    for i, col_a in art_col.items():
        tabla[-1] = tabla[-1] + tabla[i]

    idx_artificiales = list(art_col.values())

    # Encabezados
    enc_x = [f"X{j+1}" for j in range(n_vars)]
    enc_s = [f"S{i+1}" for i in range(n_rest)]
    enc_a = [f"A{k+1}" for k in range(n_art)]
    encabezados = enc_x + enc_s + enc_a + ["Sol"]

    return tabla, base, encabezados, idx_artificiales, n_vars, n_rest, n_art


# ─────────────────────────────────────────────
#  COLUMNA Y FILA PIVOTE  (igual que profe)
#  Fase I: minimizar → buscar el más POSITIVO en fila Z
#  Fase II max: buscar el más NEGATIVO
#  Fase II min: buscar el más POSITIVO (igual que fase I)
# ─────────────────────────────────────────────

def _columna_pivote_min(fila_z, excluir=None):
    """Para minimización (Fase I y Fase II min): coef más positivo."""
    n = len(fila_z)
    mejor_col = None
    mejor_val = 1e-9   # umbral
    for j in range(n):
        if excluir and j in excluir:
            continue
        if fila_z[j] > mejor_val:
            mejor_val = fila_z[j]
            mejor_col = j
    return mejor_col


def _columna_pivote_max(fila_z, excluir=None):
    """Para maximización (Fase II max): coef más negativo."""
    n = len(fila_z)
    mejor_col = None
    mejor_val = -1e-9
    for j in range(n):
        if excluir and j in excluir:
            continue
        if fila_z[j] < mejor_val:
            mejor_val = fila_z[j]
            mejor_col = j
    return mejor_col


def _fila_pivote(tabla, col):
    razones = []
    for i in range(len(tabla) - 1):
        if tabla[i, col] > 1e-9:
            razones.append(tabla[i, -1] / tabla[i, col])
        else:
            razones.append(np.inf)
    min_r = min(razones)
    if np.isinf(min_r):
        return None
    return int(np.argmin(razones))


def _pivotear(tabla, fila, col):
    tabla = tabla.copy()
    pivote = tabla[fila, col]
    tabla[fila] = tabla[fila] / pivote
    for i in range(len(tabla)):
        if i != fila:
            tabla[i] = tabla[i] - tabla[i, col] * tabla[fila]
    return tabla


# ─────────────────────────────────────────────
#  NOMBRES DE VARIABLES BÁSICAS
# ─────────────────────────────────────────────

def _nombre_base(col, n_vars, n_rest, n_art, encabezados):
    return encabezados[col]


# ─────────────────────────────────────────────
#  RESOLVER
# ─────────────────────────────────────────────

def resolver_dos_fases(form):
    tipo = form["tipo"]           # "max" o "min"
    n_vars = int(form["num_variables"])
    n_rest = int(form["num_restricciones"])

    z_coef = [float(form[f"z{j}"]) for j in range(n_vars)]

    restricciones = []
    tipos_rest = []
    for i in range(n_rest):
        fila = [float(form[f"r{i}_{j}"]) for j in range(n_vars)]
        fila.append(float(form[f"sol{i}"]))
        restricciones.append(fila)
        tipos_rest.append(form.get(f"tipo_rest{i}", "<="))

    # ── FASE I ──────────────────────────────────
    tabla, base, encabezados, idx_art, n_vars, n_rest, n_art = \
        _construir_fase1(restricciones, tipos_rest)

    iteraciones_f1 = []
    bases_f1 = []

    iteraciones_f1.append(_tabla_a_lista(tabla))
    bases_f1.append([encabezados[b] for b in base])

    max_iter = 100
    for _ in range(max_iter):
        col = _columna_pivote_min(tabla[-1, :-1])
        if col is None:
            break
        fila = _fila_pivote(tabla, col)
        if fila is None:
            break
        tabla = _pivotear(tabla, fila, col)
        base[fila] = col
        iteraciones_f1.append(_tabla_a_lista(tabla))
        bases_f1.append([encabezados[b] for b in base])

    # Verificar que Z1 ≈ 0
    z1_val = tabla[-1, -1]
    fase1_ok = abs(z1_val) < 1e-6

    if not fase1_ok:
        return {
            "error": "El problema no tiene solución factible (Z₁ ≠ 0 al finalizar Fase I)."
        }, iteraciones_f1, bases_f1, encabezados, [], [], [], encabezados

    # Valores actuales de variables básicas al final de Fase I
    vals_f1 = {}
    for j, b in enumerate(base):
        vals_f1[encabezados[b]] = _fmt(tabla[j, -1])

    # ── FASE II ─────────────────────────────────
    # Construimos nueva tabla SIN columnas de artificiales
    cols_mantener = [j for j in range(tabla.shape[1] - 1) if j not in idx_art]
    cols_mantener.append(tabla.shape[1] - 1)   # columna Sol

    tabla2 = tabla[:, cols_mantener].copy()

    # Encabezados Fase II
    enc2 = [encabezados[j] for j in cols_mantener[:-1]] + ["Sol"]

    # Reindexar base (quitar cols artificiales)
    col_map = {old: new for new, old in enumerate(cols_mantener[:-1])}
    base2 = [col_map[b] for b in base]

    # Nueva fila Z Fase II
    # Z - c1*X1 - c2*X2 ... = 0
    n_cols2 = len(enc2)  # incluye Sol
    z2_fila = np.zeros(n_cols2)
    if tipo == "max":
        for j in range(n_vars):
            z2_fila[j] = -z_coef[j]
    else:
        for j in range(n_vars):
            z2_fila[j] = z_coef[j]

    tabla2[-1] = z2_fila

    # Ajustar fila Z sumando las filas de las variables básicas actuales
    # (igual que hace la profe: Z = Z + coef * fila_basica)
    for j_orig in range(n_vars):
        j2 = col_map.get(j_orig, None)
        if j2 is None:
            continue
        coef_en_z = z2_fila[j2]
        if abs(coef_en_z) < 1e-12:
            continue
        # Encontrar si esta variable está en la base
        for fi, b in enumerate(base2):
            if b == j2:
                tabla2[-1] = tabla2[-1] - coef_en_z * tabla2[fi]
                break

    iteraciones_f2 = []
    bases_f2 = []
    iteraciones_f2.append(_tabla_a_lista(tabla2))
    bases_f2.append([enc2[b] for b in base2])

    for _ in range(max_iter):
        if tipo == "max":
            col = _columna_pivote_max(tabla2[-1, :-1])
        else:
            col = _columna_pivote_min(tabla2[-1, :-1])
        if col is None:
            break
        fila = _fila_pivote(tabla2, col)
        if fila is None:
            break
        tabla2 = _pivotear(tabla2, fila, col)
        base2[fila] = col
        iteraciones_f2.append(_tabla_a_lista(tabla2))
        bases_f2.append([enc2[b] for b in base2])

    # ── RESULTADO ───────────────────────────────
    z_raw = tabla2[-1, -1]
    z_final = _fmt(z_raw if tipo == "max" else -z_raw)

    # Valores de variables originales
    valores = {}
    for j in range(n_vars):
        col_j = col_map.get(j)
        if col_j is None:
            valores[f"X{j+1}"] = 0
            continue
        col_data = tabla2[:-1, col_j]
        ones = np.isclose(col_data, 1).sum()
        zeros = np.isclose(col_data, 0).sum()
        if ones == 1 and zeros == len(col_data) - 1:
            fi = int(np.where(np.isclose(col_data, 1))[0][0])
            valores[f"X{j+1}"] = _fmt(tabla2[fi, -1])
        else:
            valores[f"X{j+1}"] = 0

    # Holguras
    for idx_s in range(n_rest):
        col_orig = n_vars + idx_s
        col_s = col_map.get(col_orig)
        nombre = encabezados[col_orig]
        if col_s is None:
            valores[nombre] = 0
            continue
        col_data = tabla2[:-1, col_s]
        ones = np.isclose(col_data, 1).sum()
        zeros = np.isclose(col_data, 0).sum()
        if ones == 1 and zeros == len(col_data) - 1:
            fi = int(np.where(np.isclose(col_data, 1))[0][0])
            valores[nombre] = _fmt(tabla2[fi, -1])
        else:
            valores[nombre] = 0

    resultado = {
        "z": z_final,
        "tipo": tipo,
        "fase1_ok": True,
        "valores": valores
    }

    return resultado, iteraciones_f1, bases_f1, encabezados, \
           iteraciones_f2, bases_f2, enc2
