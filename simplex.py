import numpy as np


def construir_tabla(z, restricciones, tipos_restriccion, tipo_optimizacion):
    num_vars = len(z)
    num_rest = len(restricciones)

    tabla = []

    for i, (r, tipo_rest) in enumerate(zip(restricciones, tipos_restriccion)):
        fila = list(r[:-1])

        holguras = [0] * num_rest

        if tipo_rest == "<=":
            holguras[i] = 1    # variable de holgura positiva
        elif tipo_rest == ">=":
            holguras[i] = -1   # variable de exceso negativa
        elif tipo_rest == "=":
            holguras[i] = 0

        fila.extend(holguras)
        fila.append(r[-1])
        tabla.append(fila)

    if tipo_optimizacion == "max":
        # Max: fila Z con coeficientes negados → buscamos el más negativo como CP
        z_fila = [-v for v in z] + [0] * num_rest + [0]
    else:
        # Min: fila -Z con coeficientes positivos → buscamos el más positivo como CP
        z_fila = [v for v in z] + [0] * num_rest + [0]

    tabla.append(z_fila)
    return np.array(tabla, dtype=float)


def columna_pivote(tabla, tipo_optimizacion):
    if tipo_optimizacion == "max":
        col = np.argmin(tabla[-1, :-1])
        if tabla[-1, col] >= -1e-9:
            return None  # óptimo alcanzado
        return col
    else:
        col = np.argmax(tabla[-1, :-1])
        if tabla[-1, col] <= 1e-9:
            return None  # óptimo alcanzado
        return col


def fila_pivote(tabla, col):
    razones = []
    for i in range(len(tabla) - 1):
        if tabla[i, col] > 1e-9:
            razones.append(tabla[i, -1] / tabla[i, col])
        else:
            razones.append(np.inf)

    min_razon = np.min(razones)
    if np.isinf(min_razon):
        return None  # no acotado
    return int(np.argmin(razones))


def pivotear(tabla, fila, col):
    tabla = tabla.copy()
    pivote = tabla[fila, col]
    tabla[fila] = tabla[fila] / pivote
    for i in range(len(tabla)):
        if i != fila:
            tabla[i] = tabla[i] - tabla[i, col] * tabla[fila]
    return tabla


def tabla_a_lista(tabla):
    return [[round(v, 4) for v in fila] for fila in tabla]


def obtener_valores_variables(tabla, num_vars):
    valores = {}
    for j in range(num_vars):
        columna = tabla[:-1, j]
        ones = list(columna).count(1)
        zeros = list(columna).count(0)
        if ones == 1 and zeros == len(columna) - 1:
            fila = list(columna).index(1)
            valores[f"X{j+1}"] = round(tabla[fila, -1], 4)
        else:
            valores[f"X{j+1}"] = 0
    return valores


def resolver_simplex(form):
    tipo = form["tipo"]  # "max" o "min"
    n = int(form["num_variables"])
    m = int(form["num_restricciones"])

    z = [float(form[f"z{i}"]) for i in range(n)]

    restricciones = []
    tipos_restriccion = []

    for i in range(m):
        fila = [float(form[f"r{i}_{j}"]) for j in range(n)]
        fila.append(float(form[f"sol{i}"]))
        tipo_rest = form.get(f"tipo_rest{i}", "<=")
        tipos_restriccion.append(tipo_rest)
        restricciones.append(fila)

    tabla = construir_tabla(z, restricciones, tipos_restriccion, tipo)

    encabezados_holgura = []
    for i, tipo_rest in enumerate(tipos_restriccion):
        if tipo_rest == ">=":
            encabezados_holgura.append(f"S{i+1}")
        else:
            encabezados_holgura.append(f"S{i+1}")

    encabezados = [f"X{i+1}" for i in range(n)] + encabezados_holgura + ["Sol"]

    iteraciones = []
    iteraciones.append(tabla_a_lista(tabla.copy()))

    max_iter = 100
    iter_count = 0

    while iter_count < max_iter:
        col = columna_pivote(tabla, tipo)
        if col is None:
            break
        fila = fila_pivote(tabla, col)
        if fila is None:
            break
        tabla = pivotear(tabla, fila, col)
        iteraciones.append(tabla_a_lista(tabla.copy()))
        iter_count += 1

    resultado = round(tabla[-1, -1], 4)
    valores_variables = obtener_valores_variables(tabla, n)

    return resultado, iteraciones, encabezados, valores_variables