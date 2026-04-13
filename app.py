from flask import Flask, render_template, request
from grafico import resolver_grafico
from simplex import resolver_simplex

app = Flask(__name__)


@app.route("/")
def menu():
    return render_template("menu.html")


@app.route("/grafico", methods=["GET", "POST"])
def grafico():
    resultado = None
    tabla = []
    grafico_html = None

    if request.method == "POST":
        resultado, tabla, grafico_html = resolver_grafico(request.form)

    return render_template(
        "grafico.html",
        resultado=resultado,
        tabla=tabla,
        grafico=grafico_html
    )


@app.route("/simplex", methods=["GET", "POST"])
def simplex():
    resultado = None
    iteraciones = []
    encabezados = []
    valores_variables = {}

    if request.method == "POST":
        resultado, iteraciones, encabezados, valores_variables = resolver_simplex(request.form)

    return render_template(
        "simplex.html",
        resultado=resultado,
        iteraciones=iteraciones,
        encabezados=encabezados,
        valores_variables=valores_variables
    )


if __name__ == "__main__":
    app.run(debug=True)