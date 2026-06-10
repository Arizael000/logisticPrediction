# %% [markdown]
# # 📘 Cuaderno de aprendizaje: Modelos ARIMA y ARIMAX para series temporales
# 
# **Instrucciones**: Este archivo está preparado para JupyterLab. Cada celda comienza con `# %%`.  
# Ejecuta paso a paso y observa los resultados. Los comentarios explican la teoría y los parámetros clave.

# %% [markdown]
# ## 🔍 1. Introducción y conceptos fundamentales
# 
# - **Serie temporal**: datos ordenados en el tiempo (tendencia, estacionalidad, ruido).
# - **Estacionariedad**: media, varianza y autocovarianza constantes. Es requisito para ARIMA.
# - **Autocorrelación (ACF) y autocorrelación parcial (PACF)**: para identificar órdenes p y q.
# - **ARIMA(p,d,q)**: Autorregresivo (p), diferenciado (d), media móvil (q).
# - **ARIMAX**: igual pero con variables exógenas (predictores externos).

# %% [markdown]
# ## ⚙️ 2. Importación de librerías (ejecutar una vez)

# %%
# Celda de instalación (solo la primera vez, descomentar si falta algún paquete)
# !pip install statsmodels pandas matplotlib numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX  # para ARIMAX
import warnings
warnings.filterwarnings('ignore')

print("✔ Librerías cargadas correctamente")

# %% [markdown]
# ## 📊 3. Datos de ejemplo: tendencia + estacionalidad + ruido

# %%
# Generar serie temporal simulada (200 puntos mensuales)
np.random.seed(42)
n = 200
t = np.arange(n)
tendencia = 0.5 * t
estacionalidad = 10 * np.sin(2 * np.pi * t / 12)   # ciclo cada 12 meses
ruido = np.random.normal(0, 2, n)
y = tendencia + estacionalidad + ruido

# Índice de fechas para mejor manejo
fechas = pd.date_range(start='2020-01-01', periods=n, freq='M')
serie = pd.Series(y, index=fechas, name='ventas')

# Graficar
plt.figure(figsize=(12,4))
plt.plot(serie)
plt.title('Serie temporal simulada: ventas mensuales')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 🔬 4. Verificar estacionariedad (test de Dickey-Fuller Aumentado)

# %%
# H0: no estacionaria. Si p-valor < 0.05 -> estacionaria (rechazamos H0)
resultado = adfuller(serie)
print(f'Estadístico ADF: {resultado[0]:.4f}')
print(f'p-valor: {resultado[1]:.4f}')
if resultado[1] < 0.05:
    print("✔ La serie es estacionaria (podemos usar d=0)")
else:
    print("✘ La serie NO es estacionaria. Necesitamos diferenciar.")

# Observación: el p-valor será alto debido a la tendencia.

# %% [markdown]
# ## 🔁 5. Diferenciación para eliminar tendencia (determinar d)

# %%
serie_diff = serie.diff().dropna()   # diferenciación de orden 1
resultado_diff = adfuller(serie_diff)
print(f'p-valor después de diferenciar: {resultado_diff[1]:.4f}')
if resultado_diff[1] < 0.05:
    print("✔ Ahora la serie es estacionaria -> d=1")
else:
    print("⚠️ Prueba con d=2")

# Graficar la serie diferenciada
plt.figure(figsize=(12,3))
plt.plot(serie_diff)
plt.title('Serie diferenciada (d=1)')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 📈 6. Identificar p y q usando ACF y PACF

# %%
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
plot_acf(serie_diff, lags=30, ax=ax1)
plot_pacf(serie_diff, lags=30, ax=ax2)
plt.show()

# Interpretación práctica:
# - PACF con corte brusco en lag 1 → p=1 (AR(1))
# - ACF con decaimiento gradual o corte tras lag 2 → q=2 (MA(2))
# Puedes probar varias combinaciones y comparar AIC.

# %% [markdown]
# ## 🧪 7. Ajustar modelo ARIMA(p,d,q) – ejemplo (1,1,2)

# %%
modelo_arima = ARIMA(serie, order=(1,1,2))   # (p,d,q)
resultado_arima = modelo_arima.fit()
print(resultado_arima.summary())

# Parámetros clave a observar:
# - coef: estimaciones de φ (AR) y θ (MA). Deben ser significativos (P>|z| < 0.05)
# - AIC / BIC: métricas para comparar modelos (menor es mejor)
# - Ljung-Box (Q): p-valor > 0.05 indica que residuos son ruido blanco (modelo adecuado)

# %% [markdown]
# ## 🔍 8. Diagnóstico de residuos (validación del modelo)

# %%
residuos = resultado_arima.resid
fig, axes = plt.subplots(2,2, figsize=(12,6))
residuos.plot(ax=axes[0,0], title='Residuos')
axes[0,0].axhline(0, color='red', linestyle='--')
residuos.hist(ax=axes[0,1], bins=20, title='Histograma')
plot_acf(residuos, lags=20, ax=axes[1,0])

# Test de Ljung-Box para autocorrelación de residuos
lb_test = resultado_arima.test_serial_correlation(method='ljungbox', lags=10)
axes[1,1].text(0.1, 0.5, f'Ljung-Box p-valores (lags 1..10):\n{lb_test[0].flatten()}', fontsize=10)
axes[1,1].axis('off')
plt.tight_layout()
plt.show()

# Ideal: p-valores > 0.05 en todos los lags (residuos independientes)

# %% [markdown]
# ## 🔮 9. Pronóstico dentro y fuera de muestra

# %%
# Pronóstico 12 pasos adelante
forecast = resultado_arima.forecast(steps=12)
print("Pronóstico para los próximos 12 meses:")
print(forecast.values)

# Gráfico histórico + predicción
plt.figure(figsize=(12,4))
plt.plot(serie, label='Datos reales')
plt.plot(forecast, label='Pronóstico ARIMA(1,1,2)', color='red')
plt.legend()
plt.title('Predicción con ARIMA')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 🌦️ 10. ARIMAX: incluyendo una variable exógena (ejemplo)

# %%
# Simulamos una variable externa: gasto publicitario correlacionado con ventas
gasto = 20 + 0.3 * y + np.random.normal(0, 1, n)
exog = pd.Series(gasto, index=fechas, name='gasto')

# Ajustar ARIMAX (usamos SARIMAX con order y sin estacionalidad)
modelo_arimax = SARIMAX(serie, exog=exog, order=(1,1,2))
resultado_arimax = modelo_arimax.fit()
print(resultado_arimax.summary())

# Pronóstico con valores futuros de exógena (simulamos los próximos 12 meses)
exog_futuro = gasto[-12:] + np.random.normal(0, 0.5, 12)   # ejemplo burdo
forecast_arimax = resultado_arimax.forecast(steps=12, exog=exog_futuro)
print("\nPronóstico ARIMAX (12 meses):", forecast_arimax.values)

# %% [markdown]
# ## ✍️ 11. Ejercicios propuestos (para practicar)

# %% [markdown]
# ### Ejercicio 1: Selección automática de (p,q) por AIC
# Escribe una función que pruebe p=0..3 y q=0..3 con d fijo (según ADF) y elija el modelo con menor AIC.

# %%
# Solución propuesta (puedes modificarla)
def seleccionar_arima(serie, d, max_p=3, max_q=3):
    mejor_aic = np.inf
    mejor_orden = None
    for p in range(max_p+1):
        for q in range(max_q+1):
            try:
                model = ARIMA(serie, order=(p,d,q)).fit()
                aic = model.aic
                if aic < mejor_aic:
                    mejor_aic = aic
                    mejor_orden = (p,d,q)
                print(f'order=({p},{d},{q}) -> AIC={aic:.2f}')
            except:
                continue
    print(f'\nMejor modelo: ARIMA{mejor_orden} con AIC={mejor_aic:.2f}')
    return mejor_orden

# Aplicar a la serie simulada (d=1 hallado antes)
# mejor_orden = seleccionar_arima(serie, d=1)

# %% [markdown]
# ### Ejercicio 2: Validación temporal (RMSE en test)
# Divide en train (primer 80%) y test (último 20%). Ajusta ARIMA en train y predice en test. Calcula RMSE.

# %%
# Pistas:
# train = serie[:int(0.8*len(serie))]
# test = serie[int(0.8*len(serie)):]
# modelo = ARIMA(train, order=(1,1,2)).fit()
# pred = modelo.forecast(steps=len(test))
# rmse = np.sqrt(np.mean((pred - test)**2))
# print(f'RMSE fuera de muestra: {rmse:.2f}')

# %% [markdown]
# ### Ejercicio 3: Serie real (pasajeros de aerolíneas)
# - Carga datos desde `seaborn.load_dataset('flights')`
# - Descompón en tendencia, estacionalidad, residuo (`seasonal_decompose`)
# - Aplica SARIMA con componente estacional (P,D,Q,s)

# %%
# import seaborn as sns
# flights = sns.load_dataset('flights')
# # Convierte la columna 'passengers' en serie temporal mensual...
# # Luego ajusta SARIMAX con seasonal_order=(1,1,1,12)

# %% [markdown]
# ## 💡 12. Otra sugerencia adicional
# 
# Además de ARIMA/ARIMAX, considera:
# - **Prophet** (Meta): fácil, maneja festivos y cambios de tendencia.
# - **LightGBM / XGBoost** con lags y medias móviles: muy potentes en competiciones.
# - **LSTM** (redes recurrentes) para secuencias largas y no lineales.
# 
# Recomendación: domina bien ARIMA para entender los fundamentos (estacionariedad, autocorrelación, diagnóstico). Luego explora Prophet para productividad y finalmente árboles/LSTM si tienes muchos datos.
# 
# ¡Practica con los ejercicios y modifica los ejemplos!

# %% [markdown]
# ## ✅ Fin del cuaderno
# 
# Copia este bloque completo en un archivo con extensión `.py` y ábrelo en JupyterLab.  
# También puedes copiar celda por celda en un notebook nuevo. ¡A aprender!