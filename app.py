from flask import Flask, render_template, request
from grafico import resolver_grafico
from simplex import resolver_simplex
from simplex_2fases import resolver_dos_fases
from dual import resolver_dual

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


@app.route("/dos_fases", methods=["GET", "POST"])
def dos_fases():
    resultado = None
    iteraciones_f1 = []
    bases_f1 = []
    enc_f1 = []
    iteraciones_f2 = []
    bases_f2 = []
    enc_f2 = []

    if request.method == "POST":
        resultado, iteraciones_f1, bases_f1, enc_f1, \
        iteraciones_f2, bases_f2, enc_f2 = resolver_dos_fases(request.form)

    return render_template(
        "dos_fases.html",
        resultado=resultado,
        iteraciones_f1=iteraciones_f1,
        bases_f1=bases_f1,
        enc_f1=enc_f1,
        iteraciones_f2=iteraciones_f2,
        bases_f2=bases_f2,
        enc_f2=enc_f2,
    )



@app.route("/dual", methods=["GET", "POST"])
def dual():
    resultado   = None
    iteraciones = []
    bases_iter  = []
    encabezados = []

    if request.method == "POST":
        resultado, iteraciones, bases_iter, encabezados = resolver_dual(request.form)

    return render_template(
        "dual.html",
        resultado=resultado,
        iteraciones=iteraciones,
        bases_iter=bases_iter,
        encabezados=encabezados,
    )

if __name__ == "__main__":
    app.run(debug=True)