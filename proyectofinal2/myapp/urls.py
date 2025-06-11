from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
from django.contrib.auth import views as auth_views

urlpatterns = [
    #pagina raiz
    path('', views.principal, name='principal'),
    
    #paginas hijas
        #paginas del usuario
        path('myapp\templates\myapp\home.html', views.home, name='home'),
        path('myapp\templates\myapp\login.html', views.login_view, name='login'),
        path('myapp\templates\myapp\registro.html', views.registro, name='registro'),
        path('myapp\templates\myapp\creadores.html', views.creator_list, name='creator'),
        path('myapp\templates\myapp\cambio_contraseña.html', views.cambio_contraseña, name='editar'),
        path('cerrar-sesion/', views.cerrar_sesion, name='cerrar_sesion'),
        path('perfil/', views.perfil, name='perfil'),
    
        #Teoria
        path('myapp\templates\myapp\teoriaBiseccion.html', views.metodo_biseccion, name='teoriaB'),
        path('myapp\templates\myapp\teoriaDif.html', views.mostrar_teoria, name='teoriaD'),
        path('myapp\templates\myapp\punto_fijo_teoria.html', views.teoria_punto_fijo, name='teoriaPunto'),
        path('myapp\templates\myapp\biblioteca.html', views.ver_biblioteca, name='biblioteca'),
        path('myapp\templates\myapp\videos.html', views.ver_videos, name='videos'),
        path('myapp\templates\myapp\rapsonteoria.html', views.newton_raphson_theory_view, name='rapsonteoria'),
        
        #Historial del Usuario
        path('historial/', views.diferencias_historial, name='exercise_history'),
        path('biseccion_historial.html/', views.historial_biseccion, name='biseccion_history'),
        path('Historial_general.html/', views.Historial_general, name='history'),
        
        #Detalles del historial
        path('detalles_biseccion.html/<int:history_id>/', views.ver_detalle, name='detalles'),
        path('detalles_diferencias.html/<int:registro_id>/', views.detalles_registro_diferenciacion, name='detalles_diferencias'),
        path('myapp\templates\myapp\calcular_biseccion.html', views.bisection_view, name='calcular_Biseccion'),
    
        #Metodos Matematicos
        path('diferencias.html', views.diferencias, name='diferencias_divididas'),
        path('segunda_diferenciacion.html', views.segunda_diferenciacion, name='diferencias_segunda'),
        path('tercera_diferenciacion.html', views.tercera_diferenciacion, name='diferencias_tercera'),
        path('cuarta_diferenciacion.html', views.cuarta_diferenciacion, name='diferencias_cuarta'),
        path('punto_Fijo.html', views.fixed_point_method, name='Punto Fijo'),
        path('rapson.html', views.newton_raphson_view, name='rapson'),
        
        #Calculadora
        path('CalculadoraBase.html', views.calcular, name='CalculadoraBase'),

] 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
