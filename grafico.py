import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from itertools import combinations


def interseccion(a1, b1, c1, a2, b2, c2):
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([c1, c2])

    try:
        sol = np.linalg.solve(A, B)
        return sol[0], sol[1]
    except:
        return None


def cumple_restricciones(x, y, restricciones):
    if x < 0 or y < 0:
        return False

    for r in restricciones:
        valor = r["a"] * x + r["b"] * y

        if r["signo"] == "<=" and valor > r["c"] + 1e-6:
            return False

        if r["signo"] == ">=" and valor < r["c"] - 1e-6:
            return False

    return True


def obtener_vertices(restricciones):
    vertices = [(0, 0)]

    for r in restricciones:
        if r["a"] != 0:
            x = r["c"] / r["a"]
            if x >= 0:
                vertices.append((x, 0))

        if r["b"] != 0:
            y = r["c"] / r["b"]
            if y >= 0:
                vertices.append((0, y))

    for r1, r2 in combinations(restricciones, 2):
        p = interseccion(
            r1["a"], r1["b"], r1["c"],
            r2["a"], r2["b"], r2["c"]
        )

        if p:
            vertices.append(p)

    factibles = []

    for x, y in vertices:
        if cumple_restricciones(x, y, restricciones):
            factibles.append((round(x, 4), round(y, 4)))

    return list(set(factibles))


def graficar(restricciones, vertices, optimo):
    fig = go.Figure()

    x_vals = np.linspace(0, 20, 400)

    for i, r in enumerate(restricciones):
        if r["b"] != 0:
            y_vals = (r["c"] - r["a"] * x_vals) / r["b"]

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=f"R{i+1}"
            ))

        else:
            x = r["c"] / r["a"]

            fig.add_trace(go.Scatter(
                x=[x, x],
                y=[0, 20],
                mode='lines',
                name=f"R{i+1}"
            ))

    if vertices:
        vertices_ordenados = sorted(
            vertices,
            key=lambda p: np.arctan2(
                p[1] - np.mean([v[1] for v in vertices]),
                p[0] - np.mean([v[0] for v in vertices])
            )
        )

        poly_x = [p[0] for p in vertices_ordenados] + [vertices_ordenados[0][0]]
        poly_y = [p[1] for p in vertices_ordenados] + [vertices_ordenados[0][1]]

        fig.add_trace(go.Scatter(
            x=poly_x,
            y=poly_y,
            fill='toself',
            mode='lines',
            name='Región factible'
        ))

        fig.add_trace(go.Scatter(
            x=[p[0] for p in vertices],
            y=[p[1] for p in vertices],
            mode='markers+text',
            text=[f"({p[0]}, {p[1]})" for p in vertices],
            textposition="top center",
            name='Intersecciones'
        ))

    if optimo:
        fig.add_trace(go.Scatter(
            x=[optimo[0]],
            y=[optimo[1]],
            mode='markers+text',
            text=["Óptimo"],
            textposition="bottom center",
            marker=dict(size=12),
            name='Solución óptima'
        ))

    fig.update_layout(
        title="Método Gráfico Interactivo",
        xaxis_title="X1",
        yaxis_title="X2",
        dragmode='pan',
        xaxis=dict(range=[0, 20]),
        yaxis=dict(range=[0, 20]),
        template='plotly_white'
    )

    return pio.to_html(fig, full_html=False)


def resolver_grafico(form):
    tipo = form["tipo"]
    z1 = float(form["z1"])
    z2 = float(form["z2"])

    restricciones = []
    tabla = []

    n = int(form["num_restricciones"])

    for i in range(n):
        restricciones.append({
            "a": float(form[f"a{i}"]),
            "b": float(form[f"b{i}"]),
            "signo": form[f"signo{i}"],
            "c": float(form[f"c{i}"])
        })

    vertices = obtener_vertices(restricciones)

    mejor_valor = None
    mejor_punto = None

    for x, y in vertices:
        z = z1 * x + z2 * y

        tabla.append({
            "punto": f"P({x},{y})",
            "coord": f"({x}, {y})",
            "valor": round(z, 4)
        })

        if mejor_valor is None:
            mejor_valor = z
            mejor_punto = (x, y)

        else:
            if tipo == "max" and z > mejor_valor:
                mejor_valor = z
                mejor_punto = (x, y)

            elif tipo == "min" and z < mejor_valor:
                mejor_valor = z
                mejor_punto = (x, y)

    grafico_html = graficar(restricciones, vertices, mejor_punto)

    resultado = {
        "z": round(mejor_valor, 4),
        "x1": mejor_punto[0],
        "x2": mejor_punto[1]
    }

    return resultado, tabla, grafico_html