import numpy as np


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def _fmt(v):
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return round(float(v), 6)


def _tabla_a_lista(tabla):
    return [[_fmt(v) for v in fila] for fila in tabla]


# ─────────────────────────────────────────────
# CONSTRUIR FASE I
# ─────────────────────────────────────────────

def _construir_fase1(restricciones, tipos_rest):

    n_rest = len(restricciones)
    n_vars = len(restricciones[0]) - 1

    art_needed = [i for i, t in enumerate(tipos_rest) if t in (">=", "=")]
    n_art = len(art_needed)

    total_cols = n_vars + n_rest + n_art + 1

    tabla = np.zeros((n_rest + 1, total_cols))

    art_col = {}
    art_counter = 0

    base = []

    for i, (row, tipo) in enumerate(zip(restricciones, tipos_rest)):

        # Variables originales
        for j in range(n_vars):
            tabla[i, j] = row[j]

        if tipo == "<=":
            # Solo holgura
            tabla[i, n_vars + i] = 1
            base.append(n_vars + i)

        elif tipo == ">=":
            # Exceso (surplus) con signo negativo
            tabla[i, n_vars + i] = -1
            # Artificial
            col_a = n_vars + n_rest + art_counter
            tabla[i, col_a] = 1
            art_col[i] = col_a
            base.append(col_a)
            art_counter += 1

        elif tipo == "=":
            # Solo artificial (sin holgura)
            col_a = n_vars + n_rest + art_counter
            tabla[i, col_a] = 1
            art_col[i] = col_a
            base.append(col_a)
            art_counter += 1

        tabla[i, -1] = row[-1]

    # ── Construir fila W (Fase I): W = Σ artificiales ──────────────
    # Paso 1: colocar −1 en cada columna artificial (W − Σaᵢ = 0)
    for col_a in art_col.values():
        tabla[-1, col_a] = -1

    # Paso 2: eliminar variables básicas de la fila W
    # (sumar cada fila cuya variable básica es artificial)
    for i in art_col.keys():
        tabla[-1] = tabla[-1] + tabla[i]

    idx_artificiales = list(art_col.values())

    # Encabezados
    enc_x = [f"X{j+1}" for j in range(n_vars)]
    enc_s = [f"S{i+1}" for i in range(n_rest)]
    enc_a = [f"A{k+1}" for k in range(n_art)]
    encabezados = enc_x + enc_s + enc_a + ["Sol"]

    return (
        tabla,
        base,
        encabezados,
        idx_artificiales,
        n_vars,
        n_rest,
        n_art
    )


# ─────────────────────────────────────────────
# COLUMNA PIVOTE FASE I
# Para MIN W: el coeficiente más POSITIVO en la
# fila W indica la variable que más reduce W.
# ─────────────────────────────────────────────

def _columna_pivote_fase1(fila_z):

    mejor_col = None
    mejor_val = 1e-9          # umbral: ignorar valores ≈ 0

    for j in range(len(fila_z)):
        if fila_z[j] > mejor_val:
            mejor_val = fila_z[j]
            mejor_col = j

    return mejor_col


# ─────────────────────────────────────────────
# COLUMNA PIVOTE FASE II
# Para MAX (internamente): el coeficiente más
# NEGATIVO en la fila Z indica posible mejora.
# ─────────────────────────────────────────────

def _columna_pivote_fase2(fila_z):

    mejor_col = None
    mejor_val = -1e-9         # umbral: ignorar valores ≈ 0

    for j in range(len(fila_z)):
        if fila_z[j] < mejor_val:
            mejor_val = fila_z[j]
            mejor_col = j

    return mejor_col


# ─────────────────────────────────────────────
# FILA PIVOTE  (razón mínima positiva)
# ─────────────────────────────────────────────

def _fila_pivote(tabla, col):

    razones = []

    for i in range(len(tabla) - 1):
        if tabla[i, col] > 1e-9:
            razones.append(tabla[i, -1] / tabla[i, col])
        else:
            razones.append(np.inf)

    menor = min(razones)

    if np.isinf(menor):
        return None

    return int(np.argmin(razones))


# ─────────────────────────────────────────────
# PIVOTEAR  (eliminación de Gauss-Jordan)
# ─────────────────────────────────────────────

def _pivotear(tabla, fila, col):

    tabla = tabla.copy()
    pivote = tabla[fila, col]
    tabla[fila] = tabla[fila] / pivote

    for i in range(len(tabla)):
        if i != fila:
            tabla[i] = tabla[i] - tabla[i, col] * tabla[fila]

    return tabla


# ─────────────────────────────────────────────
# RESOLVER DOS FASES
# ─────────────────────────────────────────────

def resolver_dos_fases(form):

    tipo   = form["tipo"]                         # "max" | "min"
    n_vars = int(form["num_variables"])
    n_rest = int(form["num_restricciones"])

    z_coef = [float(form[f"z{j}"]) for j in range(n_vars)]

    restricciones = []
    tipos_rest    = []

    for i in range(n_rest):
        fila = [float(form[f"r{i}_{j}"]) for j in range(n_vars)]
        fila.append(float(form[f"sol{i}"]))
        restricciones.append(fila)
        tipos_rest.append(form.get(f"tipo_rest{i}", "<="))

    # ═════════════════════════════════════════
    # FASE I  — minimizar W = Σ artificiales
    # ═════════════════════════════════════════

    (
        tabla,
        base,
        encabezados,
        idx_art,
        n_vars,
        n_rest,
        n_art
    ) = _construir_fase1(restricciones, tipos_rest)

    iteraciones_f1 = []
    bases_f1       = []

    iteraciones_f1.append(_tabla_a_lista(tabla))
    bases_f1.append([encabezados[b] for b in base])

    max_iter = 100

    for _ in range(max_iter):

        col = _columna_pivote_fase1(tabla[-1, :-1])
        if col is None:
            break

        fila = _fila_pivote(tabla, col)
        if fila is None:
            return (
                {"error": "Problema no acotado en Fase I"},
                [], [], [], [], [], []
            )

        tabla = _pivotear(tabla, fila, col)
        base[fila] = col

        iteraciones_f1.append(_tabla_a_lista(tabla))
        bases_f1.append([encabezados[b] for b in base])

    # ── Verificar factibilidad ────────────────
    w_final = tabla[-1, -1]

    if abs(w_final) > 1e-6:
        return (
            {"error": "El problema no tiene solución factible."},
            iteraciones_f1, bases_f1, encabezados, [], [], []
        )

    # ═════════════════════════════════════════
    # TRANSICIÓN: eliminar columnas artificiales
    # ═════════════════════════════════════════

    cols_mantener = [
        j for j in range(tabla.shape[1] - 1)
        if j not in idx_art
    ] + [tabla.shape[1] - 1]

    tabla2 = tabla[:, cols_mantener].copy()
    enc2   = [encabezados[j] for j in cols_mantener[:-1]] + ["Sol"]

    col_map = {
        old: new
        for new, old in enumerate(cols_mantener[:-1])
    }

    # Reconstruir base usando solo variables no-artificiales
    base2 = [col_map[b] for b in base if b in col_map]

    # ═════════════════════════════════════════
    # FASE II — optimizar el objetivo original
    # ═════════════════════════════════════════
    #
    # Convención interna: siempre se trabaja como MAX.
    #
    #   MAX Z  →  fila Z = −c_j  (negar coeficientes)
    #             Optimal cuando todos los c̄_j ≥ 0
    #             z_raw = Z_max  →  z_final = z_raw
    #
    #   MIN Z  →  equivale a MAX(−Z)
    #             fila Z = +c_j  (NO negar)
    #             Optimal cuando todos los c̄_j ≥ 0
    #             z_raw = −Z_min  →  z_final = −z_raw
    #
    # ─────────────────────────────────────────
    z2 = np.zeros(len(enc2))

    for j in range(n_vars):
        if tipo == "max":
            # MAX Z: colocar −c_j para que buscar "más negativo" = mejorar Z
            z2[j] = -z_coef[j]
        else:
            # MIN Z → MAX(−Z): colocar +c_j
            # El ajuste posterior puede dejar coefs negativos → se pivotea normalmente
            z2[j] = z_coef[j]

    tabla2[-1] = z2

    # Ajustar fila Z eliminando variables básicas (Gaussian para objetivo)
    for fi, b in enumerate(base2):
        coef = tabla2[-1, b]
        if abs(coef) > 1e-9:
            tabla2[-1] = tabla2[-1] - coef * tabla2[fi]

    iteraciones_f2 = []
    bases_f2       = []

    iteraciones_f2.append(_tabla_a_lista(tabla2))
    bases_f2.append([enc2[b] for b in base2])

    for _ in range(max_iter):

        col = _columna_pivote_fase2(tabla2[-1, :-1])
        if col is None:
            break

        fila = _fila_pivote(tabla2, col)
        if fila is None:
            return (
                {"error": "Problema no acotado en Fase II"},
                [], [], [], [], [], []
            )

        tabla2 = _pivotear(tabla2, fila, col)
        base2[fila] = col

        iteraciones_f2.append(_tabla_a_lista(tabla2))
        bases_f2.append([enc2[b] for b in base2])

    # ═════════════════════════════════════════
    # EXTRAER SOLUCIÓN
    # ═════════════════════════════════════════

    valores = {}

    # Variables de decisión
    for j in range(n_vars):
        col_data = tabla2[:-1, j]
        ones  = np.isclose(col_data, 1).sum()
        zeros = np.isclose(col_data, 0).sum()

        if ones == 1 and zeros == len(col_data) - 1:
            fila = np.where(np.isclose(col_data, 1))[0][0]
            valores[f"X{j+1}"] = _fmt(tabla2[fila, -1])
        else:
            valores[f"X{j+1}"] = 0

    # Variables de holgura / exceso
    for s in range(n_rest):
        col = n_vars + s
        if col >= tabla2.shape[1] - 1:
            continue

        col_data = tabla2[:-1, col]
        ones  = np.isclose(col_data, 1).sum()
        zeros = np.isclose(col_data, 0).sum()
        nombre = enc2[col]

        if ones == 1 and zeros == len(col_data) - 1:
            fila = np.where(np.isclose(col_data, 1))[0][0]
            valores[nombre] = _fmt(tabla2[fila, -1])
        else:
            valores[nombre] = 0

    # ── Verificar restricciones ───────────────
    tol = 1e-6

    for i, rest in enumerate(restricciones):

        lhs = sum(rest[j] * valores[f"X{j+1}"] for j in range(n_vars))
        rhs = rest[-1]
        tipo_r = tipos_rest[i]

        if tipo_r == "<=":
            factible = lhs <= rhs + tol
        elif tipo_r == ">=":
            factible = lhs >= rhs - tol
        else:  # "="
            factible = abs(lhs - rhs) <= tol

        if not factible:
            return (
                {"error": f"La restricción {i+1} no se cumple: "
                           f"LHS={round(lhs,6)}, RHS={rhs}, tipo={tipo_r}"},
                iteraciones_f1, bases_f1, encabezados,
                iteraciones_f2, bases_f2, enc2
            )

    # ── Calcular valor óptimo ─────────────────
    z_raw = tabla2[-1, -1]

    if tipo == "min":
        # z_raw almacena −Z_min (porque trabajamos MAX(−Z))
        # Por tanto  Z_min = −z_raw
        z_final = _fmt(-z_raw)
    else:
        # z_raw almacena directamente Z_max
        z_final = _fmt(z_raw)

    resultado = {
        "z":        z_final,
        "tipo":     tipo,
        "fase1_ok": True,
        "valores":  valores
    }

    return (
        resultado,
        iteraciones_f1,
        bases_f1,
        encabezados,
        iteraciones_f2,
        bases_f2,
        enc2
    )


# ─────────────────────────────────────────────
# TESTS RÁPIDOS
# ─────────────────────────────────────────────

if __name__ == "__main__":

    sep = "─" * 55

    # ── Test 1: MAX con restricción de igualdad ──────────────
    # Max Z = 2x1 + 3x2
    # x1 + x2  = 4
    # x1 + 3x2 ≤ 6
    # Óptimo esperado: x1=3, x2=1, Z=9
    form1 = {
        "tipo": "max",
        "num_variables": 2,
        "num_restricciones": 2,
        "z0": 2, "z1": 3,
        "r0_0": 1, "r0_1": 1, "sol0": 4, "tipo_rest0": "=",
        "r1_0": 1, "r1_1": 3, "sol1": 6, "tipo_rest1": "<=",
    }

    res1, *_ = resolver_dos_fases(form1)
    print(sep)
    print("Test 1 — MAX Z = 2x1 + 3x2  |  x1+x2=4, x1+3x2≤6")
    if "error" in res1:
        print("  ERROR:", res1["error"])
    else:
        print(f"  Z    = {res1['z']}   (esperado 9)")
        print(f"  vars = {res1['valores']}")

    # ── Test 2: MIN con restricciones ≥ ──────────────────────
    # Min Z = 4x1 + 6x2
    # 2x1 + x2  ≥ 4
    #  x1 + 2x2 ≥ 4
    # Óptimo esperado: x1=4/3, x2=4/3, Z=40/3≈13.333
    form2 = {
        "tipo": "min",
        "num_variables": 2,
        "num_restricciones": 2,
        "z0": 4, "z1": 6,
        "r0_0": 2, "r0_1": 1, "sol0": 4, "tipo_rest0": ">=",
        "r1_0": 1, "r1_1": 2, "sol1": 4, "tipo_rest1": ">=",
    }

    res2, *_ = resolver_dos_fases(form2)
    print(sep)
    print("Test 2 — MIN Z = 4x1 + 6x2  |  2x1+x2≥4, x1+2x2≥4")
    if "error" in res2:
        print("  ERROR:", res2["error"])
    else:
        print(f"  Z    = {res2['z']}   (esperado {round(40/3, 6)})")
        print(f"  vars = {res2['valores']}")

    # ── Test 3: MIN con objetivo negativo ─────────────────────
    # Min Z = -x1 - x2
    # x1 ≤ 3
    # x2 ≤ 2
    # x1 + x2 ≥ 1   (fuerza Fase I)
    # Óptimo esperado: x1=3, x2=2, Z=-5
    form3 = {
        "tipo": "min",
        "num_variables": 2,
        "num_restricciones": 3,
        "z0": -1, "z1": -1,
        "r0_0": 1, "r0_1": 0, "sol0": 3, "tipo_rest0": "<=",
        "r1_0": 0, "r1_1": 1, "sol1": 2, "tipo_rest1": "<=",
        "r2_0": 1, "r2_1": 1, "sol2": 1, "tipo_rest2": ">=",
    }

    res3, *_ = resolver_dos_fases(form3)
    print(sep)
    print("Test 3 — MIN Z = -x1 - x2  |  x1≤3, x2≤2, x1+x2≥1")
    if "error" in res3:
        print("  ERROR:", res3["error"])
    else:
        print(f"  Z    = {res3['z']}   (esperado -5)")
        print(f"  vars = {res3['valores']}")

    # ── Test 4: MAX con restricciones ≥ y ≤ mixtas ───────────
    # Max Z = 5x1 + 4x2
    # 6x1 + 4x2 ≤ 24
    #  x1 + 2x2  = 6
    # Óptimo esperado: x1=3, x2=3/2, Z=21
    form4 = {
        "tipo": "max",
        "num_variables": 2,
        "num_restricciones": 2,
        "z0": 5, "z1": 4,
        "r0_0": 6, "r0_1": 4, "sol0": 24, "tipo_rest0": "<=",
        "r1_0": 1, "r1_1": 2, "sol1":  6, "tipo_rest1": "=",
    }

    res4, *_ = resolver_dos_fases(form4)
    print(sep)
    print("Test 4 — MAX Z = 5x1 + 4x2  |  6x1+4x2≤24, x1+2x2=6")
    if "error" in res4:
        print("  ERROR:", res4["error"])
    else:
        print(f"  Z    = {res4['z']}   (esperado 21)")
        print(f"  vars = {res4['valores']}")

    # ── Test 5: infactible ────────────────────────────────────
    # x1 + x2 ≤ 1  y  x1 + x2 ≥ 3  → sin solución
    form5 = {
        "tipo": "max",
        "num_variables": 2,
        "num_restricciones": 2,
        "z0": 1, "z1": 1,
        "r0_0": 1, "r0_1": 1, "sol0": 1, "tipo_rest0": "<=",
        "r1_0": 1, "r1_1": 1, "sol1": 3, "tipo_rest1": ">=",
    }

    res5, *_ = resolver_dos_fases(form5)
    print(sep)
    print("Test 5 — Infactible  |  x1+x2≤1 y x1+x2≥3")
    if "error" in res5:
        print("  Error detectado correctamente:", res5["error"])
    else:
        print("  FALLO: debería haber reportado infactibilidad")

    print(sep)