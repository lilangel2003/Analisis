########################################################################################################################################################################

#Librerias
from django.shortcuts import render,redirect
import math
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
from django.http import JsonResponse
from fractions import Fraction
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from .forms import *
from django.contrib.auth import update_session_auth_hash
import sympy as sp
from .models import *
from sympy import *
import csv
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm
from sympy import symbols, lambdify
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from django.http import HttpResponseForbidden
import io
import os
import urllib, base64
import pandas as pd
from django.contrib import messages
from .models import *
from django.contrib.auth.models import User
from .models import Usuarios
from .models import DifferenceDividedHistory
from django.conf import settings

########################################################################################################################################################################

#Vista principal
def principal(request):
    return render(request ,'myapp/index.html')

########################################################################################################################################################################

def home(request):
    return render(request ,'myapp/index.html')

########################################################################################################################################################################


def newton_raphson_theory_view(request):
    theory = {
        'introduccion': """
        El método de Newton-Raphson es una técnica iterativa utilizada para encontrar las raíces de una función. Se basa en una aproximación lineal de la función en torno a un punto inicial y utiliza esta aproximación para mejorar la estimación de la raíz.
        """,
        'formula': """
        La fórmula general del método de Newton-Raphson es:
        $$ x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)} $$
        """,
        'proceso': """
        <ul>
            <li>Seleccione una estimación inicial $$ x_0 $$.</li>
            <li>Calcule la siguiente estimación utilizando la fórmula del método.</li>
            <li>Reemplace $$ x_n $$ por la nueva estimación y repita el proceso hasta que la diferencia entre estimaciones consecutivas sea menor que un umbral de tolerancia.</li>
        </ul>
        """,
        'ejemplo': """
        Supongamos que queremos encontrar una raíz de la función $$ f(x) = x^2 - 2 $$. La derivada de esta función es $$ f'(x) = 2x $$.
        
        Si comenzamos con una estimación inicial de $$ x_0 = 1.0 $$, la primera iteración sería:
        $$ x_1 = x_0 - \\frac{f(x_0)}{f'(x_0)} = 1.0 - \\frac{1^2 - 2}{2*1} = 1.0 - \\frac{-1}{2} = 1.5 $$
        En la siguiente iteración, repetimos el proceso usando $$ x_1 $$ como nueva estimación.
        """
    }
    return render(request, 'myapp/rapsonteoria.html', {'theory': theory})


#Vista del login y validaciones
def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            correo_User = form.cleaned_data['username']
            password_User = form.cleaned_data['password']
            # Autenticacion
            user = authenticate(request, username=correo_User, password=password_User)
            if user is not None:
                login(request, user)
                return redirect('principal')
            else:
                error_message = f"Usuario: {correo_User}, Contraseña: {password_User}"
                form.add_error(None, error_message)
    else:
        form = LoginForm()

    return render(request, 'myapp/login.html', {'form': form})

########################################################################################################################################################################

def teoria_punto_fijo(request):
    teoria = """
    <h2>Introducción al Método de Punto Fijo:</h2>
    <p>El método de punto fijo es un método iterativo utilizado para encontrar soluciones de ecuaciones de la forma:</p>
    <pre>x = g(x)</pre>
    <p>Donde:</p>
    <ul>
        <li><strong>x</strong>: Es la variable para la cual se desea encontrar una solución.</li>
        <li><strong>g(x)</strong>: Es una función continua que transforma la variable <strong>x</strong>.</li>
    </ul>

    <h2>Fórmula Iterativa:</h2>
    <p>La fórmula general para el método de punto fijo es:</p>
    <pre>x_{n+1} = g(x_n)</pre>
    <p>Donde:</p>
    <ul>
        <li><strong>x_{n+1}</strong>: Es el nuevo valor calculado.</li>
        <li><strong>x_n</strong>: Es el valor de la iteración anterior.</li>
    </ul>

    <h2>Convergencia:</h2>
    <p>Para que el método de punto fijo converja, la función <strong>g(x)</strong> debe cumplir con ciertas condiciones, como:</p>
    <ul>
        <li>La función <strong>g(x)</strong> debe ser continua en el intervalo de interés.</li>
        <li>El valor absoluto de la derivada de la función <strong>g(x)</strong> debe ser menor que 1 en el punto fijo.</li>
    </ul>

    <p>Matemáticamente, esto se expresa como:</p>
    <pre>|g'(x)| < 1</pre>
    <p>Donde:</p>
    <ul>
        <li><strong>g'(x)</strong>: Es la derivada de la función <strong>g(x)</strong>.</li>
    </ul>
    """
    return render(request, 'myapp/punto_fijo_teoria.html', {'teoria': teoria})


#Funcion para la teoria de Diferenciacion Numerica
def teoria_diferencias():
    teoria = """
    <h2>Diferencia hacia adelante:</h2>
    <p>La derivada hacia adelante se aproxima utilizando la siguiente fórmula:</p>
    <pre>f'(x) ≈ (f(x + h) - f(x)) / h</pre>
    <p>Donde:</p>
    <ul>
        <li>f(x): Valor de la función en el punto x.</li>
        <li>f(x + h): Valor de la función en el punto x + h.</li>
        <li>h: Tamaño del paso.</li>
    </ul>

    <h2>Diferencia hacia atrás:</h2>
    <p>La derivada hacia atrás se aproxima utilizando la siguiente fórmula:</p>
    <pre>f'(x) ≈ (f(x) - f(x - h)) / h</pre>
    <p>Donde:</p>
    <ul>
        <li>f(x): Valor de la función en el punto x.</li>
        <li>f(x - h): Valor de la función en el punto x - h.</li>
        <li>h: Tamaño del paso.</li>
    </ul>

    <h2>Diferencia central:</h2>
    <p>La derivada central se aproxima utilizando la siguiente fórmula:</p>
    <pre>f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h)</pre>
    <p>Donde:</p>
    <ul>
        <li>f(x + h): Valor de la función en el punto x + h.</li>
        <li>f(x - h): Valor de la función en el punto x - h.</li>
        <li>h: Tamaño del paso.</li>
    </ul>
    """
    return teoria

########################################################################################################################################################################

def mostrar_teoria(request):
    contexto = {
        'explicacion_teorica': teoria_diferencias()
    }
    return render(request, 'myapp/teoriaDif.html', contexto)

########################################################################################################################################################################
#Funcion de la teoria de Biseccion
def metodo_biseccion(request):
    formulas = [
        {
            'titulo': 'Punto Medio del Intervalo',
            'formula': 'c = (a + b) / 2',
            'explicacion': 'Donde a y b son los extremos del intervalo inicial, y c es el punto medio.'
        },
        {
            'titulo': 'Criterio de Convergencia',
            'formula': 'f(c) = 0',
            'explicacion': 'El método de bisección determina si la raíz se encuentra en el intervalo izquierdo [a, c] o derecho [c, b] según el cambio de signo en la función evaluada en c.'
        },
        {
            'titulo': 'Actualización del Intervalo',
            'formula': 'Dependiendo del criterio de convergencia, se actualiza el intervalo de búsqueda para la siguiente iteración.',
            'explicacion': 'Si f(a) * f(c) < 0, se actualiza el intervalo a [a, c]. Si f(c) * f(b) < 0, se actualiza el intervalo a [c, b].'
        }
    ]
    
    explicacion_teorica = """
    En el método de bisección se utilizan varias fórmulas para iterar y encontrar la raíz de una ecuación dentro de un intervalo dado. 
    Las principales fórmulas que se emplean en este método son las siguientes:

    1. **Punto Medio del Intervalo**:
       c = (a + b) / 2
       donde a y b son los extremos del intervalo inicial, y c es el punto medio.

    2. **Criterio de Convergencia**:
       El método de bisección determina si la raíz se encuentra en el intervalo izquierdo [a, c] o derecho [c, b] según el cambio de signo en la función evaluada en c:
       - Si f(c) = 0, entonces c es la raíz.
       - Si f(a) * f(c) < 0, la raíz está en el intervalo [a, c].
       - Si f(c) * f(b) < 0, la raíz está en el intervalo [c, b].

    3. **Actualización del Intervalo**:
       Dependiendo del criterio de convergencia, se actualiza el intervalo de búsqueda para la siguiente iteración:
       - Si f(a) * f(c) < 0, se actualiza el intervalo a [a, c].
       - Si f(c) * f(b) < 0, se actualiza el intervalo a [c, b].

    Estas fórmulas y criterios son fundamentales para el funcionamiento del método de bisección, que es un método numérico básico pero efectivo para encontrar raíces de ecuaciones no lineales dentro de un intervalo dado.
    """

    context = {
        'formulas': formulas,
        'explicacion_teorica': explicacion_teorica,
    }
    
    return render(request, 'myapp/teoriaBiseccion.html', context)

########################################################################################################################################################################

# Función auxiliar para encontrar el intervalo inicial
def find_initial_interval(equation, x, x_start, x_end, step):
    f = sp.lambdify(x, equation, 'numpy')
    x_values = np.arange(x_start, x_end + step, step)
    y_values = f(x_values)
    
    intervals = []
    for i in range(len(x_values) - 1):
        if np.sign(y_values[i]) != np.sign(y_values[i + 1]):
            intervals.append((round(x_values[i], 4), round(x_values[i + 1], 4)))
    
    if len(intervals) == 0:
        return None
    elif len(intervals) > 1:
        return intervals[0]
    else:
        return intervals[0]

########################################################################################################################################################################

# Función auxiliar para el método de bisección
def bisection_method(a, b, tol, f):
    if f(a) * f(b) >= 0:
        return None, "El método de bisección no garantiza convergencia en este intervalo."
    
    iter_count = 0
    while True:
        c = (a + b) / 2
        
        if abs(f(c)) < tol:
            return round(c, 4), f"¡Se encontró la raíz aproximada dentro de la tolerancia! x = {round(c, 4)}"
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        
        iter_count += 1
    
    return round((a + b) / 2, 4), "Iteraciones completadas. La aproximación final de la raíz es x = {round((a + b) / 2, 4)}"

########################################################################################################################################################################

# Función auxiliar para generar las iteraciones
def generate_iterations(a, b, tol, f):
    iteraciones = []
    iter_count = 0
    previous_c = None
    
    while True:
        c = (a + b) / 2
        error = abs((c - previous_c) / c) if previous_c is not None else None
        
        iteraciones.append((iter_count, round(f(c), 4), round(c, 4), round(error, 4) if error is not None else None))
        
        if abs(f(c)) < tol:
            break
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        
        previous_c = c
        iter_count += 1
    
    return iteraciones

########################################################################################################################################################################

# Función auxiliar para generar una gráfica
def generate_plot(equation_str, x_start, x_end, raiz):
    # Crear un símbolo para la variable x
    x = sp.symbols('x')
    
    # Convertir la ecuación a un objeto simbólico
    equation = sp.sympify(equation_str)
    
    # Convertir la ecuación en una función numérica usando lambdify
    f = sp.lambdify(x, equation, 'numpy')
    
    # Crear datos para la gráfica
    x_values = np.linspace(x_start, x_end, 400)
    y_values = f(x_values)
    
    # Generar la gráfica usando matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Función')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(raiz, color='red', linestyle='--', label='Raíz')
    plt.scatter(raiz, f(raiz), color='red')
    plt.title('Gráfica de la función y raíz encontrada')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    
    # Convertir la gráfica a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    grafica_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return grafica_base64

########################################################################################################################################################################

def bisection_view(request):
    mensaje = None
    resultado_biseccion = None
    grafica_base64 = None
    
    if request.method == 'POST':
        form = BiseccionForm(request.POST)
        if form.is_valid():

            equation_str = form.cleaned_data['equation']
            x_start = form.cleaned_data['x_start']
            x_end = form.cleaned_data['x_end']
            step = form.cleaned_data['step']
            tol = form.cleaned_data['tol']
            
            if x_start > x_end:
                mensaje = "El valor inicial del intervalo debe ser menor que el valor final."
            else:
                
                x = sp.symbols('x')
                
                equation = sp.sympify(equation_str)
                
                interval = find_initial_interval(equation, x, x_start, x_end, step)
                if interval is None:
                    mensaje = "No se encontraron cambios de signo en el intervalo dado."
                else:
                    
                    f = sp.lambdify(x, equation, 'numpy')
                    
                    
                    raiz, mensaje_raiz = bisection_method(interval[0], interval[1], tol, f)
                    
                    if raiz is not None:

                        iteraciones = generate_iterations(interval[0], interval[1], tol, f)
                        iteraciones.append((raiz, None)) 
                        
                        resultado_biseccion = (raiz, len(iteraciones) - 1, None, iteraciones, mensaje_raiz, f)
                        
                        grafica_base64 = generate_plot(equation_str, x_start, x_end, raiz)
                        
                        if request.user.is_authenticated:
                            BiseccionHistory.objects.create(
                                user=request.user,
                                ecuacion=equation_str,
                                valor_min=x_start,
                                valor_max=x_end,
                                error_porcentual=tol,
                                raiz_aproximada=raiz,
                                iter_count=len(iteraciones) - 1,
                            )
                        
                    else:
                        mensaje = "El método de bisección no garantiza convergencia en este intervalo."
        
        else:
            mensaje = "Formulario inválido. Por favor, revise los datos ingresados."
    
    else:
        form = BiseccionForm()
    
    return render(request, 'myapp/calcular_biseccion.html', {
        'form': form,
        'mensaje': mensaje,
        'resultado_biseccion': resultado_biseccion,
        'grafica_base64': grafica_base64,
    })

########################################################################################################################################################################

# Función para calcular la derivada numérica usando diferencia hacia adelante
def derivada_forward(f, x, h):
    f_x = round(f(x), 4)
    f_x_plus_h = round(f(x + h), 4)
    resultado = round((f_x_plus_h - f_x) / h, 4)
    formula_sustituida = f"({f_x_plus_h} - {f_x}) / {h}"
    pasos = [
        f"f(x) = {f_x}",
        f"f(x + h) = {f_x_plus_h}",
        f"({f_x_plus_h} - {f_x}) / {h} = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

########################################################################################################################################################################

# Función para calcular la derivada numérica usando diferencia hacia atras
def derivada_backward(f, x, h):
    f_x = round(f(x), 4)
    f_x_minus_h = round(f(x - h), 4)
    resultado = round((f_x - f_x_minus_h) / h, 4)
    formula_sustituida = f"({f_x} - {f_x_minus_h}) / {h}"
    pasos = [
        f"f(x) = {f_x}",
        f"f(x - h) = {f_x_minus_h}",
        f"({f_x} - {f_x_minus_h}) / {h} = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

########################################################################################################################################################################

# Función para calcular la derivada numérica usando diferencia central
def derivada_central(f, x, h):
    f_x = round(f(x), 4)
    f_x_plus_h = round(f(x + h), 4)
    f_x_minus_h = round(f(x - h), 4)
    resultado = round((f_x_plus_h - f_x_minus_h) / (2 * h), 4)
    formula_sustituida = f"({f_x_plus_h} - {f_x_minus_h}) / (2 * {h})"
    pasos = [
        f"f(x) = {f_x}",
        f"f(x + h) = {f_x_plus_h}",
        f"f(x - h) = {f_x_minus_h}",
        f"({f_x_plus_h} - {f_x_minus_h}) / (2 * {h}) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

########################################################################################################################################################################

#Calcula error de las diferencias
def calcular_errores(derivada_fwd, derivada_bwd, derivada_cen, derivada_exacta_val):
    error_fwd = abs(derivada_fwd - derivada_exacta_val)
    error_bwd = abs(derivada_bwd - derivada_exacta_val)
    error_cen = abs(derivada_cen - derivada_exacta_val)
    
    return {
        'error_fwd': round(error_fwd, 4),
        'error_bwd': round(error_bwd, 4),
        'error_cen': round(error_cen, 4)
    }

########################################################################################################################################################################

#Funcion del metodo de diferenciacion numerica
def diferencias(request):
    resultado = {}
    grafica_url = ""
    
    if request.method == 'POST':
        form = DiferenciacionForm(request.POST)
        if form.is_valid():
            funcion = form.cleaned_data['f']
            valor_x = float(form.cleaned_data['x'])
            valor_h = float(form.cleaned_data['h'])

            try:
                x = symbols('x')
                ecuacion = sympify(funcion)
                f = lambdify(x, ecuacion)
                
                derivada_exacta = diff(ecuacion, x)
                derivada_exacta_func = lambdify(x, derivada_exacta)
                derivada_exacta_val = round(derivada_exacta_func(valor_x), 4)

                derivada_fwd, formula_fwd, pasos_fwd = derivada_forward(f, valor_x, valor_h)
                derivada_bwd, formula_bwd, pasos_bwd = derivada_backward(f, valor_x, valor_h)
                derivada_cen, formula_cen, pasos_cen = derivada_central(f, valor_x, valor_h)

                errores = calcular_errores(derivada_fwd, derivada_bwd, derivada_cen, derivada_exacta_val)

                if request.user.is_authenticated:
                    DifferenceDividedHistory.objects.create(
                        user=request.user,
                        function=funcion,
                        x_value=valor_x,
                        h_value=valor_h,
                        derivada_fwd=derivada_fwd,
                        derivada_bwd=derivada_bwd,
                        derivada_cen=derivada_cen,
                        derivada_exacta=derivada_exacta_val,
                        error_fwd=errores['error_fwd'],
                        error_bwd=errores['error_bwd'],
                        error_cen=errores['error_cen']
                    )

                resultado = {
                    'derivada_fwd': derivada_fwd,
                    'formula_fwd': formula_fwd,
                    'pasos_fwd': pasos_fwd,
                    'derivada_bwd': derivada_bwd,
                    'formula_bwd': formula_bwd,
                    'pasos_bwd': pasos_bwd,
                    'derivada_cen': derivada_cen,
                    'formula_cen': formula_cen,
                    'pasos_cen': pasos_cen,
                    'derivada_exacta': derivada_exacta_val,
                    'error_fwd': round(errores['error_fwd'], 4),
                    'error_bwd': round(errores['error_bwd'], 4),
                    'error_cen': round(errores['error_cen'], 4)
                }
                
                x_vals = [valor_x - 2*valor_h, valor_x - valor_h, valor_x, valor_x + valor_h, valor_x + 2*valor_h]
                y_vals = [f(val) for val in x_vals]
                
                plt.figure()
                plt.plot(x_vals, y_vals, 'b-', label='Función')
                plt.plot(valor_x, f(valor_x), 'ro', label='Punto de Evaluación')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.title('Gráfica de la Función y Puntos Encontrados')
                plt.legend()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                string = base64.b64encode(buf.read())
                grafica_url = 'data:image/png;base64,' + urllib.parse.quote(string)
                buf.close()

                messages.success(request, 'Cálculo de derivadas completado y guardado correctamente.')

            except ValueError as e:
                messages.error(request, f'Ocurrió un error de valor: {str(e)}')
            except SyntaxError as e:
                messages.error(request, f'Ocurrió un error de sintaxis en la ecuación: {str(e)}')
            except Exception as e:
                messages.error(request, f'Ocurrió un error durante el cálculo: {str(e)}')

    else:
        form = DiferenciacionForm()

    return render(request, 'myapp/diferencias.html', {'form': form, 'resultado': resultado, 'grafica_url': grafica_url})

########################################################################################################################################################################

# Función para calcular la segunda derivada numérica usando diferencia hacia adelante
def segunda_derivada_forward(f, x, h):
    f_x = round(f(x), 4)
    f_x_plus_h = round(f(x + h), 4)
    f_x_plus_2h = round(f(x + 2 * h), 4)
    resultado = round((f_x_plus_2h - 2 * f_x_plus_h + f_x) / (h ** 2), 4)
    formula_sustituida = f"({f_x_plus_2h} - 2 * {f_x_plus_h} + {f_x}) / ({h}^2)"
    pasos = [
        f"f(x) = {f_x}",
        f"f(x + h) = {f_x_plus_h}",
        f"f(x + 2h) = {f_x_plus_2h}",
        f"({f_x_plus_2h} - 2 * {f_x_plus_h} + {f_x}) / ({h}^2) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

# Función para calcular la segunda derivada numérica usando diferencia hacia atrás
def segunda_derivada_backward(f, x, h):
    f_x = round(f(x), 4)
    f_x_minus_h = round(f(x - h), 4)
    f_x_minus_2h = round(f(x - 2 * h), 4)
    resultado = round((f_x - 2 * f_x_minus_h + f_x_minus_2h) / (h ** 2), 4)
    formula_sustituida = f"({f_x} - 2 * {f_x_minus_h} + {f_x_minus_2h}) / ({h}^2)"
    pasos = [
        f"f(x) = {f_x}",
        f"f(x - h) = {f_x_minus_h}",
        f"f(x - 2h) = {f_x_minus_2h}",
        f"({f_x} - 2 * {f_x_minus_h} + {f_x_minus_2h}) / ({h}^2) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

# Función para calcular la segunda derivada numérica usando diferencia central
def segunda_derivada_central(f, x, h):
    f_x_plus_h = round(f(x + h), 4)
    f_x_minus_h = round(f(x - h), 4)
    f_x = round(f(x), 4)
    resultado = round((f_x_plus_h - 2 * f_x + f_x_minus_h) / (h ** 2), 4)
    formula_sustituida = f"({f_x_plus_h} - 2 * {f_x} + {f_x_minus_h}) / ({h}^2)"
    pasos = [
        f"f(x + h) = {f_x_plus_h}",
        f"f(x) = {f_x}",
        f"f(x - h) = {f_x_minus_h}",
        f"({f_x_plus_h} - 2 * {f_x} + {f_x_minus_h}) / ({h}^2) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

# Vista para manejar la diferenciación de segundo orden
def segunda_diferenciacion(request):
    resultado = {}
    grafica_url = ""

    if request.method == 'POST':
        form = DiferenciacionForm(request.POST)
        if form.is_valid():
            funcion = form.cleaned_data['f']
            valor_x = float(form.cleaned_data['x'])
            valor_h = float(form.cleaned_data['h'])

            try:
                x = symbols('x')
                ecuacion = sympify(funcion)
                f = lambdify(x, ecuacion)

                # Calcular las segundas derivadas numéricas
                forward_resultado, forward_formula, forward_pasos = segunda_derivada_forward(f, valor_x, valor_h)
                backward_resultado, backward_formula, backward_pasos = segunda_derivada_backward(f, valor_x, valor_h)
                central_resultado, central_formula, central_pasos = segunda_derivada_central(f, valor_x, valor_h)

                resultado = {
                    'forward_resultado': forward_resultado,
                    'forward_formula': forward_formula,
                    'forward_pasos': forward_pasos,
                    'backward_resultado': backward_resultado,
                    'backward_formula': backward_formula,
                    'backward_pasos': backward_pasos,
                    'central_resultado': central_resultado,
                    'central_formula': central_formula,
                    'central_pasos': central_pasos
                }

                # Gráfica
                x_vals = [valor_x - 2*valor_h, valor_x - valor_h, valor_x, valor_x + valor_h, valor_x + 2*valor_h]
                y_vals = [f(val) for val in x_vals]

                plt.figure()
                plt.plot(x_vals, y_vals, 'b-', label='Función')
                plt.plot(valor_x, f(valor_x), 'ro', label='Punto de Evaluación')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.title('Gráfica de la Función y Puntos Encontrados')
                plt.legend()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)

                string = base64.b64encode(buf.read())
                grafica_url = 'data:image/png;base64,' + urllib.parse.quote(string)
                buf.close()

                messages.success(request, 'Cálculo de segundas derivadas completado correctamente.')

            except ValueError as e:
                messages.error(request, f'Ocurrió un error de valor: {str(e)}')
            except SyntaxError as e:
                messages.error(request, f'Ocurrió un error de sintaxis en la ecuación: {str(e)}')
            except Exception as e:
                messages.error(request, f'Ocurrió un error durante el cálculo: {str(e)}')

    else:
        form = DiferenciacionForm()

    return render(request, 'myapp/segunda_diferenciacion.html', {'form': form, 'resultado': resultado, 'grafica_url': grafica_url})

########################################################################################################################################################################

# Función para calcular la tercera derivada numérica usando diferencia hacia adelante
def tercera_derivada_forward(f, x, h):
    f_x = round(f(x), 4)
    f_x_plus_h = round(f(x + h), 4)
    f_x_plus_2h = round(f(x + 2 * h), 4)
    f_x_plus_3h = round(f(x + 3 * h), 4)
    resultado = round((f_x_plus_3h - 3 * f_x_plus_2h + 3 * f_x_plus_h - f_x) / (h ** 3), 4)
    formula_sustituida = f"({f_x_plus_3h} - 3 * {f_x_plus_2h} + 3 * {f_x_plus_h} - {f_x}) / ({h}^3)"
    pasos = [
        f"f(x) = {f_x}",
        f"f(x + h) = {f_x_plus_h}",
        f"f(x + 2h) = {f_x_plus_2h}",
        f"f(x + 3h) = {f_x_plus_3h}",
        f"({f_x_plus_3h} - 3 * {f_x_plus_2h} + 3 * {f_x_plus_h} - {f_x}) / ({h}^3) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

# Función para calcular la tercera derivada numérica usando diferencia hacia atrás
def tercera_derivada_backward(f, x, h):
    f_x = round(f(x), 4)
    f_x_minus_h = round(f(x - h), 4)
    f_x_minus_2h = round(f(x - 2 * h), 4)
    f_x_minus_3h = round(f(x - 3 * h), 4)
    resultado = round((f_x - 3 * f_x_minus_h + 3 * f_x_minus_2h - f_x_minus_3h) / (h ** 3), 4)
    formula_sustituida = f"({f_x} - 3 * {f_x_minus_h} + 3 * {f_x_minus_2h} - {f_x_minus_3h}) / ({h}^3)"
    pasos = [
        f"f(x) = {f_x}",
        f"f(x - h) = {f_x_minus_h}",
        f"f(x - 2h) = {f_x_minus_2h}",
        f"f(x - 3h) = {f_x_minus_3h}",
        f"({f_x} - 3 * {f_x_minus_h} + 3 * {f_x_minus_2h} - {f_x_minus_3h}) / ({h}^3) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

# Función para calcular la tercera derivada numérica usando diferencia central
def tercera_derivada_central(f, x, h):
    f_x_plus_h = round(f(x + h), 4)
    f_x_minus_h = round(f(x - h), 4)
    f_x_plus_2h = round(f(x + 2 * h), 4)
    f_x_minus_2h = round(f(x - 2 * h), 4)
    resultado = round((f_x_minus_2h - 2 * f_x_minus_h + 2 * f_x_plus_h - f_x_plus_2h) / (2 * h ** 3), 4)
    formula_sustituida = f"({f_x_minus_2h} - 2 * {f_x_minus_h} + 2 * {f_x_plus_h} - {f_x_plus_2h}) / (2 * {h}^3)"
    pasos = [
        f"f(x + h) = {f_x_plus_h}",
        f"f(x + 2h) = {f_x_plus_2h}",
        f"f(x - h) = {f_x_minus_h}",
        f"f(x - 2h) = {f_x_minus_2h}",
        f"({f_x_minus_2h} - 2 * {f_x_minus_h} + 2 * {f_x_plus_h} - {f_x_plus_2h}) / (2 * {h}^3) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

# Vista para manejar la diferenciación de tercer orden
def tercera_diferenciacion(request):
    resultado = {}
    grafica_url = ""

    if request.method == 'POST':
        form = DiferenciacionForm(request.POST)
        if form.is_valid():
            funcion = form.cleaned_data['f']
            valor_x = float(form.cleaned_data['x'])
            valor_h = float(form.cleaned_data['h'])

            try:
                x = symbols('x')
                ecuacion = sympify(funcion)
                f = lambdify(x, ecuacion)

                # Calcular las terceras derivadas numéricas
                forward_resultado, forward_formula, forward_pasos = tercera_derivada_forward(f, valor_x, valor_h)
                backward_resultado, backward_formula, backward_pasos = tercera_derivada_backward(f, valor_x, valor_h)
                central_resultado, central_formula, central_pasos = tercera_derivada_central(f, valor_x, valor_h)

                resultado = {
                    'forward_resultado': forward_resultado,
                    'forward_formula': forward_formula,
                    'forward_pasos': forward_pasos,
                    'backward_resultado': backward_resultado,
                    'backward_formula': backward_formula,
                    'backward_pasos': backward_pasos,
                    'central_resultado': central_resultado,
                    'central_formula': central_formula,
                    'central_pasos': central_pasos
                }

                # Gráfica
                x_vals = [valor_x - 2*valor_h, valor_x - valor_h, valor_x, valor_x + valor_h, valor_x + 2*valor_h]
                y_vals = [f(val) for val in x_vals]

                plt.figure()
                plt.plot(x_vals, y_vals, 'b-', label='Función')
                plt.plot(valor_x, f(valor_x), 'ro', label='Punto de Evaluación')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.title('Gráfica de la Función y Puntos Encontrados')
                plt.legend()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)

                string = base64.b64encode(buf.read())
                grafica_url = 'data:image/png;base64,' + urllib.parse.quote(string)
                buf.close()

                messages.success(request, 'Cálculo de terceras derivadas completado correctamente.')

            except ValueError as e:
                messages.error(request, f'Ocurrió un error de valor: {str(e)}')
            except SyntaxError as e:
                messages.error(request, f'Ocurrió un error de sintaxis en la ecuación: {str(e)}')
            except Exception as e:
                messages.error(request, f'Ocurrió un error durante el cálculo: {str(e)}')

    else:
        form = DiferenciacionForm()

    return render(request, 'myapp/tercera_diferenciacion.html', {'form': form, 'resultado': resultado, 'grafica_url': grafica_url})

########################################################################################################################################################################

# Función para calcular la cuarta derivada numérica usando diferencia hacia adelante
def cuarta_derivada_forward(f, x, h):
    f_x = round(f(x), 4)
    f_x_plus_h = round(f(x + h), 4)
    f_x_plus_2h = round(f(x + 2 * h), 4)
    f_x_plus_3h = round(f(x + 3 * h), 4)
    f_x_plus_4h = round(f(x + 4 * h), 4)
    resultado = round((-f_x_plus_4h + 4*f_x_plus_3h - 6*f_x_plus_2h + 4*f_x_plus_h - f_x) / (h ** 4), 4)
    formula_sustituida = f"(-{f_x_plus_4h} + 4 * {f_x_plus_3h} - 6 * {f_x_plus_2h} + 4 * {f_x_plus_h} - {f_x}) / ({h}^4)"
    pasos = [
        f"f(x) = {f_x}",
        f"f(x + h) = {f_x_plus_h}",
        f"f(x + 2h) = {f_x_plus_2h}",
        f"f(x + 3h) = {f_x_plus_3h}",
        f"f(x + 4h) = {f_x_plus_4h}",
        f"(-{f_x_plus_4h} + 4 * {f_x_plus_3h} - 6 * {f_x_plus_2h} + 4 * {f_x_plus_h} - {f_x}) / ({h}^4) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

# Función para calcular la cuarta derivada numérica usando diferencia hacia atrás
def cuarta_derivada_backward(f, x, h):
    f_x = round(f(x), 4)
    f_x_minus_h = round(f(x - h), 4)
    f_x_minus_2h = round(f(x - 2 * h), 4)
    f_x_minus_3h = round(f(x - 3 * h), 4)
    f_x_minus_4h = round(f(x - 4 * h), 4)
    resultado = round((-f_x_minus_4h + 4*f_x_minus_3h - 6*f_x_minus_2h + 4*f_x_minus_h - f_x) / (h ** 4), 4)
    formula_sustituida = f"(-{f_x_minus_4h} + 4 * {f_x_minus_3h} - 6 * {f_x_minus_2h} + 4 * {f_x_minus_h} - {f_x}) / ({h}^4)"
    pasos = [
        f"f(x) = {f_x}",
        f"f(x - h) = {f_x_minus_h}",
        f"f(x - 2h) = {f_x_minus_2h}",
        f"f(x - 3h) = {f_x_minus_3h}",
        f"f(x - 4h) = {f_x_minus_4h}",
        f"(-{f_x_minus_4h} + 4 * {f_x_minus_3h} - 6 * {f_x_minus_2h} + 4 * {f_x_minus_h} - {f_x}) / ({h}^4) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

# Función para calcular la cuarta derivada numérica usando diferencia central
def cuarta_derivada_central(f, x, h):
    f_x_plus_2h = round(f(x + 2 * h), 4)
    f_x_plus_h = round(f(x + h), 4)
    f_x_minus_h = round(f(x - h), 4)
    f_x_minus_2h = round(f(x - 2 * h), 4)
    resultado = round((f_x_minus_2h - 4*f_x_minus_h + 6*f(x) - 4*f_x_plus_h + f_x_plus_2h) / (h ** 4), 4)
    formula_sustituida = f"({f_x_minus_2h} - 4 * {f_x_minus_h} + 6 * {f(x)} - 4 * {f_x_plus_h} + {f_x_plus_2h}) / ({h}^4)"
    pasos = [
        f"f(x - 2h) = {f_x_minus_2h}",
        f"f(x - h) = {f_x_minus_h}",
        f"f(x) = {f(x)}",
        f"f(x + h) = {f_x_plus_h}",
        f"f(x + 2h) = {f_x_plus_2h}",
        f"({f_x_minus_2h} - 4 * {f_x_minus_h} + 6 * {f(x)} - 4 * {f_x_plus_h} + {f_x_plus_2h}) / ({h}^4) = {resultado}"
    ]
    return resultado, formula_sustituida, pasos

# Vista para manejar la diferenciación de cuarto orden
def cuarta_diferenciacion(request):
    resultado = {}
    grafica_url = ""

    if request.method == 'POST':
        form = DiferenciacionForm(request.POST)
        if form.is_valid():
            funcion = form.cleaned_data['f']
            valor_x = float(form.cleaned_data['x'])
            valor_h = float(form.cleaned_data['h'])

            try:
                x = symbols('x')
                ecuacion = sympify(funcion)
                f = lambdify(x, ecuacion)

                # Calcular las cuartas derivadas numéricas
                forward_resultado, forward_formula, forward_pasos = cuarta_derivada_forward(f, valor_x, valor_h)
                backward_resultado, backward_formula, backward_pasos = cuarta_derivada_backward(f, valor_x, valor_h)
                central_resultado, central_formula, central_pasos = cuarta_derivada_central(f, valor_x, valor_h)

                resultado = {
                    'forward_resultado': forward_resultado,
                    'forward_formula': forward_formula,
                    'forward_pasos': forward_pasos,
                    'backward_resultado': backward_resultado,
                    'backward_formula': backward_formula,
                    'backward_pasos': backward_pasos,
                    'central_resultado': central_resultado,
                    'central_formula': central_formula,
                    'central_pasos': central_pasos
                }

                # Gráfica
                x_vals = [valor_x - 2*valor_h, valor_x - valor_h, valor_x, valor_x + valor_h, valor_x + 2*valor_h]
                y_vals = [f(val) for val in x_vals]

                plt.figure()
                plt.plot(x_vals, y_vals, 'b-', label='Función')
                plt.plot(valor_x, f(valor_x), 'ro', label='Punto de Evaluación')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.title('Gráfica de la Función y Puntos Encontrados')
                plt.legend()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)

                string = base64.b64encode(buf.read())
                grafica_url = 'data:image/png;base64,' + urllib.parse.quote(string)
                buf.close()

                messages.success(request, 'Cálculo de cuartas derivadas completado correctamente.')

            except ValueError as e:
                messages.error(request, f'Ocurrió un error de valor: {str(e)}')
            except SyntaxError as e:
                messages.error(request, f'Ocurrió un error de sintaxis en la ecuación: {str(e)}')
            except Exception as e:
                messages.error(request, f'Ocurrió un error durante el cálculo: {str(e)}')

    else:
        form = DiferenciacionForm()

    return render(request, 'myapp/cuarta_diferenciacion.html', {'form': form, 'resultado': resultado, 'grafica_url': grafica_url})

########################################################################################################################################################################

#Funcion de nuevo registro de usuario
def registro(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        form = RegistroForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()

            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)

            if user is not None:
                login(request, user)
                messages.success(request, f'Bienvenido {username}, tu cuenta ha sido creada exitosamente.')
                return redirect('home')
            else:
                messages.error(request, 'Hubo un problema al autenticarse. Inténtelo de nuevo.')
        else:
            messages.error(request, 'Error al crear la cuenta. Verifique los datos ingresados.')
    else:
        form = RegistroForm()

    return render(request, 'myapp/registro.html', {'form': form})

########################################################################################################################################################################

#Funcion del cambio de contraseña
def cambio_contraseña(request):
    mensaje = ""
    tipo_alerta = ""

    if request.method == 'POST':
        form = CambioContraseñaForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, '¡Contraseña cambiada exitosamente!')
            return redirect('home')
        else:
            messages.error(request, 'Ha ocurrido un error. Por favor, verifica los datos ingresados.')
    else:
        form = CambioContraseñaForm(request.user)

    return render(request, 'myapp/cambio_contraseña.html', {'form': form, 'mensaje': mensaje, 'tipo_alerta': tipo_alerta})

########################################################################################################################################################################

#Funcion de cerrar secion
def cerrar_sesion(request):
    logout(request)
    return redirect('/')

########################################################################################################################################################################

#Vista para ver el perfil de usuario
@login_required
def perfil(request):
    user = request.user
    return render(request, 'myapp/perfil.html', {'user': user})

########################################################################################################################################################################

#Funcion para imprimir el proceso
def generar_pdf(request):
    resultado_biseccion = request.GET.get('resultado_biseccion')

    if not resultado_biseccion:
        return HttpResponse("No se encontraron resultados para generar el PDF.", content_type="text/plain")

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="resultado_biseccion.pdf"'

    c = canvas.Canvas(response, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 40, "Resultados de la Bisección")

    c.drawString(100, height - 60, f"Raíz aproximada: {resultado_biseccion}")
    c.drawString(100, height - 80, "Número de iteraciones: No calculado")
    c.drawString(100, height - 100, "Error: No calculado")

    c.showPage()
    c.save()

    return response

########################################################################################################################################################################

#Funcion para filtrar el historial del usuario
@login_required
def diferencias_historial(request):
    history = DifferenceDividedHistory.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'myapp/historial.html', {'history': history})

########################################################################################################################################################################

# Función para calcular la derivada numérica usando diferencia hacia atras
@login_required
def historial_biseccion(request):
    history = BiseccionHistory.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'myapp/biseccion_historial.html', {'history': history})

########################################################################################################################################################################

# Función para mostrar el historial de los dos metodos
@login_required
def Historial_general(request):
    history = BiseccionHistory.objects.filter(user=request.user).order_by('-created_at')
    historydiferencias = DifferenceDividedHistory.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'myapp/Historial_general.html', {'history': history, 'historydiferencias': historydiferencias})

########################################################################################################################################################################

# Función para mostrar los libros de biblioteca
def ver_biblioteca(request):
    json_path = os.path.join(settings.BASE_DIR, 'libros.json')
    with open(json_path, 'r') as file:
        libros = json.load(file)
    return render(request, 'myapp/biblioteca.html', {'libros': libros})

########################################################################################################################################################################

# Función para mostrar los videos de biblioteca
@login_required
def ver_videos(request):
    json_path = os.path.join(settings.BASE_DIR, 'videos.json')
    with open(json_path, 'r') as file:
        videos = json.load(file)
    return render(request, 'myapp/videos.html', {'videos': videos})

########################################################################################################################################################################

#Funcion para mostrar los desarrolladores del groyecto
def creator_list(request):
    return render(request, 'myapp/creadores.html')

########################################################################################################################################################################

def ver_detalle(request, history_id):
    history_obj = get_object_or_404(BiseccionHistory, id=history_id, user=request.user)
    resultado_biseccion = None
    grafica_base64 = None
    mensaje = None
    
    if request.method == 'POST':
        form = BiseccionForm(request.POST)
        if form.is_valid():
            equation_str = form.cleaned_data['equation']
            x_start = form.cleaned_data['x_start']
            x_end = form.cleaned_data['x_end']
            step = form.cleaned_data['step']
            tol = form.cleaned_data['tol']
            
            if x_start > x_end:
                mensaje = "El valor inicial del intervalo debe ser menor que el valor final."
            else:
                x = sp.symbols('x')
                
                equation = sp.sympify(equation_str)
                
                interval = find_initial_interval(equation, x, x_start, x_end, step)
                if interval is None:
                    mensaje = "No se encontraron cambios de signo en el intervalo dado."
                else:
                    f = sp.lambdify(x, equation, 'numpy')

                    raiz, mensaje_raiz = bisection_method(interval[0], interval[1], tol, f)
                    
                    if raiz is not None:
                        # Preparar resultados
                        iteraciones = generate_iterations(interval[0], interval[1], tol, f)
                        iteraciones.append((raiz, None))
                        
                        resultado_biseccion = (raiz, len(iteraciones) - 1, None, iteraciones, mensaje_raiz, f)

                        grafica_base64 = generate_plot(equation_str, x_start, x_end, raiz)
                        
                    else:
                        mensaje = "El método de bisección no garantiza convergencia en este intervalo."
        
        else:
            mensaje = "Formulario inválido. Por favor, revise los datos ingresados."
    
    else:
        form_data = {
            'equation': history_obj.ecuacion,
            'x_start': history_obj.valor_min,
            'x_end': history_obj.valor_max,
            'tol': history_obj.error_porcentual,
        }
        form = BiseccionForm(initial=form_data)
    
    return render(request, 'myapp/detalles_biseccion.html', {
        'form': form,
        'history_obj': history_obj,
        'resultado_biseccion': resultado_biseccion,
        'grafica_base64': grafica_base64,
        'mensaje': mensaje,
    })
    return render(request, 'myapp/detalles_biseccion.html', {'form': form, 'history_obj': history_obj})

########################################################################################################################################################################

def detalles_registro_diferenciacion(request, registro_id):
    history_dif = get_object_or_404(DifferenceDividedHistory, id=registro_id, user=request.user)

    if request.method == 'POST':
        form = DiferenciacionForm(request.POST)
        if form.is_valid():
            funcion = form.cleaned_data['f']
            valor_x = float(form.cleaned_data['x'])
            valor_h = float(form.cleaned_data['h'])

            try:
                x = symbols('x')
                ecuacion = sympify(funcion)
                f = lambdify(x, ecuacion)
                
                derivada_exacta = diff(ecuacion, x)
                derivada_exacta_func = lambdify(x, derivada_exacta)
                derivada_exacta_val = round(derivada_exacta_func(valor_x), 4)

                derivada_fwd, formula_fwd, pasos_fwd = derivada_forward(f, valor_x, valor_h)
                derivada_bwd, formula_bwd, pasos_bwd = derivada_backward(f, valor_x, valor_h)
                derivada_cen, formula_cen, pasos_cen = derivada_central(f, valor_x, valor_h)

                errores = calcular_errores(derivada_fwd, derivada_bwd, derivada_cen, derivada_exacta_val)

                resultado = {
                    'derivada_fwd': derivada_fwd,
                    'formula_fwd': formula_fwd,
                    'pasos_fwd': pasos_fwd,
                    'derivada_bwd': derivada_bwd,
                    'formula_bwd': formula_bwd,
                    'pasos_bwd': pasos_bwd,
                    'derivada_cen': derivada_cen,
                    'formula_cen': formula_cen,
                    'pasos_cen': pasos_cen,
                    'derivada_exacta': derivada_exacta_val,
                    'error_fwd': round(errores['error_fwd'], 4),
                    'error_bwd': round(errores['error_bwd'], 4),
                    'error_cen': round(errores['error_cen'], 4)
                }
                
                x_vals = [valor_x - 2*valor_h, valor_x - valor_h, valor_x, valor_x + valor_h, valor_x + 2*valor_h]
                y_vals = [f(val) for val in x_vals]
                
                plt.figure()
                plt.plot(x_vals, y_vals, 'b-', label='Función')
                plt.plot(valor_x, f(valor_x), 'ro', label='Punto de Evaluación')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.title('Gráfica de la Función y Puntos Encontrados')
                plt.legend()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                string = base64.b64encode(buf.read())
                grafica_url = 'data:image/png;base64,' + urllib.parse.quote(string)
                buf.close()

                messages.success(request, 'Cálculo de derivadas completado y guardado correctamente.')

            except ValueError as e:
                messages.error(request, f'Ocurrió un error de valor: {str(e)}')
            except SyntaxError as e:
                messages.error(request, f'Ocurrió un error de sintaxis en la ecuación: {str(e)}')
            except Exception as e:
                messages.error(request, f'Ocurrió un error durante el cálculo: {str(e)}')

    else:
        form = DiferenciacionForm(initial={
            'f': history_dif.function,
            'x': history_dif.x_value,
            'h': history_dif.h_value
        })

    return render(request, 'myapp/detalles_diferencias.html', {
        'form': form,
        'history_obj': history_dif,
        'grafica_url': grafica_url if 'grafica_url' in locals() else "",
        'resultado': resultado if 'resultado' in locals() else None
    })

########################################################################################################################################################################

def calcular(request):
    resultado = None

    if request.method == 'POST':
        expression = request.POST.get('expression', '')

        try:
            # Evaluar logaritmo natural y logaritmo de base específica
            if 'ln(' in expression:
                expression = expression.replace('ln(', 'sp.ln(')
                resultado = eval(expression, {}, {'sp': sp})
            elif 'log(' in expression:
                base_end_index = expression.find(')')
                base = expression[expression.find('log(') + 4:base_end_index]
                expression = expression[base_end_index+1:]
                expression = expression.replace('log(', f'sp.log({base}, ')
                resultado = eval(expression, {}, {'sp': sp})
            elif 'pow2(' in expression:
                expression = expression.replace('pow2(', '**2')
                resultado = eval(expression)
            elif 'solve(' in expression:
                # Procesar sistemas de ecuaciones
                expression = expression.replace('solve(', '').replace(')', '')
                eqs = expression.split(';')
                symbols_list = list('xyz')  # Definir las variables usadas en las ecuaciones
                eqs = [sp.Eq(eval(eq.strip(), {}, {'x': sp.Symbol('x'), 'y': sp.Symbol('y'), 'z': sp.Symbol('z')}), 0) for eq in eqs]
                resultado = sp.solve(eqs, symbols_list)
            elif 'matrix(' in expression:
                # Procesar matrices
                expression = expression.replace('matrix(', '').replace(')', '')
                rows = expression.split(';')
                matrix_data = [list(map(float, row.split(','))) for row in rows]
                matrix = sp.Matrix(matrix_data)
                resultado = {
                    'transpose': matrix.T,
                    'inverse': matrix.inv() if matrix.det() != 0 else 'No invertible'
                }
            else:
                # Evaluar expresiones básicas y trigonométricas
                resultado = eval(expression, {}, {'sp': sp})

        except Exception as e:
            resultado = f"Error: {str(e)}"
    return render(request, 'myapp/CalculadoraBase.html', {'resultado': resultado})

def evaluate_function(expression, x):
    x_sym = symbols('x')
    transformations = (standard_transformations + (implicit_multiplication_application,))
    f_expr = parse_expr(expression, transformations=transformations)
    f = lambdify(x_sym, f_expr, 'numpy')
    return f(x)

def find_interval(f, x_start, x_end, step):
    x = x_start
    while x < x_end:
        if f(x) * f(x + step) < 0:
            return x, x + step
        x += step
    
    print(f"No se encontró un intervalo que contenga una raíz en el rango {x_start} a {x_end} con un paso de {step}.")
    return None, None

def calculate_fixed_point(f, initial_guess, tolerance, max_iterations):
    x = symbols('x')
    x0 = initial_guess
    error = float('inf')
    iterations = 0
    results = []

    while error > tolerance and iterations < max_iterations:
        x1 = f(x0)
        error = abs(x1 - x0)
        
        # Obtener la fórmula original y la sustitución
        formula_substitution = f"x0 = {x0:.4f} ⇒ f(x0) = {x1:.4f} (Sustitución)"
        
        # Redondear los valores a 4 decimales
        x0_rounded = round(x0, 4)
        x1_rounded = round(x1, 4)
        error_rounded = round(error, 4)
        
        results.append((iterations, formula_substitution, x0_rounded, x1_rounded, error_rounded))
        x0 = x1
        iterations += 1

    final_result = round(x0, 4)
    return final_result, results, iterations

def fixed_point_method(request):
    context = {}
    if request.method == 'POST':
        form = FixedPointForm(request.POST)
        if form.is_valid():
            function_expr = form.cleaned_data['equation']
            x_start = form.cleaned_data['x_start']
            x_end = form.cleaned_data['x_end']
            step = form.cleaned_data['step']
            tolerance = form.cleaned_data['tol']

            try:
                if x_start >= x_end:
                    raise ValueError("El valor de x_start debe ser menor que x_end.")
                
                if step <= 0:
                    raise ValueError("El valor de step debe ser positivo.")

                f = lambda x: evaluate_function(function_expr, x)

                interval_start, interval_end = find_interval(f, x_start, x_end, step)
                
                if interval_start is None or interval_end is None:
                    raise ValueError("No se encontró un intervalo que contenga una raíz.")
                
                initial_guess = (interval_start + interval_end) / 2
                final_result, results, iterations = calculate_fixed_point(f, initial_guess, tolerance, 100)

                context = {
                    'form': form,
                    'results': results,
                    'final_result': final_result,
                    'iterations': iterations,
                    'tolerance': tolerance,
                    'interval_start': interval_start,
                    'interval_end': interval_end
                }
            except Exception as e:
                context = {
                    'form': form,
                    'error': str(e)
                }
        else:
            context['form'] = form
    else:
        form = FixedPointForm()
        context['form'] = form

    return render(request, 'myapp/punto_Fijo.html', context)



def newton_raphson_method(f, df, x0, tol):
    iter_count = 0
    x_n = x0

    while True:
        fx_n = f(x_n)
        dfx_n = df(x_n)
        
        if dfx_n == 0:
            return None, "La derivada se volvió cero. El método falla."
        
        x_n1 = x_n - fx_n / dfx_n
        
        if abs(x_n1 - x_n) < tol:
            return round(x_n1, 4), f"¡Se encontró la raíz aproximada dentro de la tolerancia! x = {round(x_n1, 4)}"
        
        x_n = x_n1
        iter_count += 1

def generate_newton_iterations(f, df, x0, tol):
    iteraciones = []
    x_n = x0
    iter_count = 0
    
    while True:
        fx_n = f(x_n)
        dfx_n = df(x_n)
        
        if dfx_n == 0:
            break
        
        x_n1 = x_n - fx_n / dfx_n
        error = abs(x_n1 - x_n)
        
        iteraciones.append((iter_count, round(fx_n, 4), round(dfx_n, 4), round(x_n1, 4), round(error, 4)))
        
        if error < tol:
            break
        
        x_n = x_n1
        iter_count += 1
    
    return iteraciones

def generate_newton_plot(equation_str, x_start, x_end, raiz):
    x = sp.symbols('x')
    equation = sp.sympify(equation_str)
    f = sp.lambdify(x, equation, 'numpy')
    
    x_values = np.linspace(x_start, x_end, 400)
    y_values = f(x_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Función')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(raiz, color='red', linestyle='--', label='Raíz')
    plt.scatter(raiz, f(raiz), color='red')
    plt.title('Gráfica de la función y raíz encontrada')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    grafica_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return grafica_base64

def newton_raphson_view(request):
    mensaje = None
    resultado_newton = None
    grafica_base64 = None
    
    if request.method == 'POST':
        form = NewtonRaphsonForm(request.POST)
        if form.is_valid():
            equation_str = form.cleaned_data['equation']
            x0 = form.cleaned_data['x0']
            tol = form.cleaned_data['tol']
            
            x = sp.symbols('x')
            equation = sp.sympify(equation_str)
            f = sp.lambdify(x, equation, 'numpy')
            df = sp.lambdify(x, sp.diff(equation, x), 'numpy')
            
            raiz, mensaje_raiz = newton_raphson_method(f, df, x0, tol)
            
            if raiz is not None:
                iteraciones = generate_newton_iterations(f, df, x0, tol)
                resultado_newton = (raiz, len(iteraciones), iteraciones, mensaje_raiz)
                grafica_base64 = generate_newton_plot(equation_str, x0-1, x0+1, raiz)
            else:
                mensaje = mensaje_raiz
        else:
            mensaje = "Formulario inválido. Por favor, revise los datos ingresados."
    else:
        form = NewtonRaphsonForm()
    
    return render(request, 'myapp/rapson.html', {
        'form': form,
        'mensaje': mensaje,
        'resultado_newton': resultado_newton,
        'grafica_base64': grafica_base64,
    })