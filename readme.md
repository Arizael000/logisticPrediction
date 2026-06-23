# Forecasting Logístico con Prophet

Predicción de demanda a 90 días para 7 categorías de productos alimenticios, optimizando inventario y evitando overstock en una empresa de distribución de alimentos.

---

## Problema de Negocio

La empresa compraba inventario sin un forecast cuantitativo. En el mes 1 de 2023, el inventario (2,002 uds) superó las ventas reales (756 uds) en un **165%** — 1,246 unidades de capital inmovilizado. Dado que los productos son perecederos (sin retorno posible por vencimiento), el overstock representa pérdida directa.

**Objetivo:** Predecir la demanda diaria por categoría en un horizonte de 90 días, permitiendo compras basadas en datos en lugar de promedios históricos.

---

## Pipeline

```
1.BusinessAnalysis.ipynb   → Definición del problema y alcance
2.PreCleaning.ipynb        → Limpieza y validación de datos
3.EDA.ipynb                → Análisis exploratorio
7.WeeklyForecast.ipynb     → Entrenamiento de modelos Prophet
4.Visualizations.ipynb     → Visualizaciones generales (8 gráficos)
5.ClientReport.ipynb       → Reporte final para el cliente
```

---

## Dataset

| Ítem | Valor |
|------|-------|
| Registros raw | 28,358 |
| Registros limpios | 22,640 |
| Período histórico | 2023-01-03 → 2024-12-11 (603 días) |
| Productos | 10 |
| Categorías | 7 (Bebidas, Carnes, Congelados, Descartables, Embutidos, Enlatados, Lácteos) |
| Outliers detectados (IQR) | 799 — mantenidos (picos legítimos en festivos) |

**Archivos de datos:**
- `ventas_empresa_logistica.csv` — Datos crudos originales
- `clean_data.csv` — Datos limpios (22,640 registros, 10 columnas)

---

## Metodología

**Modelo:** Prophet (Meta/Facebook) — seleccionado sobre ARIMA por:
- Manejo robusto de estacionalidad semanal
- Tolerancia a outliers y datos faltantes
- Intervalos de confianza bayesianos (MCMC) no negativos
- Detección adaptativa de changepoints

**Entrenamiento:**
- Agregación semanal (W-MON) para reducir ruido
- Expansión a diario usando perfiles históricos de día de semana
- Grid search: changepoint_prior_scale ∈ [0.001, 0.01, 0.05, 0.1] × yearly_seasonality ∈ [False, 3]
- Validación cruzada con 2 ventanas rodantes
- Métrica de evaluación: SMAPE

**Resultados — Prophet vs ARIMA (SMAPE):**

| Categoría | ARIMA | Prophet v3 | Mejora |
|-----------|-------|------------|--------|
| Bebidas | 48.7% | 46.6% | +2.1% |
| Carnes | 41.3% | 35.9% | +5.4% |
| Congelados | 57.5% | 43.4% | +14.1% |
| Descartables | 43.9% | 32.3% | +11.6% |
| Embutidos | 68.6% | 35.7% | +32.9% |
| Enlatados | 54.1% | 45.8% | +8.3% |
| Lácteos | 42.5% | 32.3% | +10.2% |
| **Promedio** | **50.9%** | **38.9%** | **+12.1%** |

---

## Resultados del Forecast (90 días)

| Categoría | Demanda Pronosticada | Recomendación | Confianza |
|-----------|---------------------|---------------|-----------|
| Descartables | 4,814 uds | Compra estándar | Alta |
| Bebidas | 2,174 uds | Compra estándar | Alta |
| Lácteos | 2,119 uds | Compra estándar | Alta |
| Carnes | 1,333 uds | Compra estándar | Alta |
| Enlatados | 1,305 uds | Compra estándar | Alta |
| Congelados | 819 uds | Compra estándar | Alta |
| Embutidos | 328 uds | Compra estándar | Alta |

**Ahorro por overstock evitable:**
- Compra tradicional estimada: 12,328 uds
- Compra con forecast: 12,891 uds (con margen de seguridad)
- Overstock evitable: ~230 uds (1.9%)
- **Advertencia:** Intervalos de confianza amplios — se recomienda margen de seguridad del 20-30%

---

## Archivos Generados

### Forecast (CSV)
| Archivo | Descripción |
|---------|-------------|
| `forecast_consolidado_90d.csv` | Forecast consolidado (637 filas, todas las categorías) |
| `forecast_Bebidas_90d.csv` | Forecast individual — Bebidas |
| `forecast_Carnes_90d.csv` | Forecast individual — Carnes |
| `forecast_Congelados_90d.csv` | Forecast individual — Congelados |
| `forecast_Descartables_90d.csv` | Forecast individual — Descartables |
| `forecast_Embutidos_90d.csv` | Forecast individual — Embutidos |
| `forecast_Enlatados_90d.csv` | Forecast individual — Enlatados |
| `forecast_Lacteos_90d.csv` | Forecast individual — Lácteos |
| `resumen_forecast_categorias.csv` | Resumen estadístico por categoría |
| `powerbi_forecast_completo.csv` | Datos históricos + forecast combinados para PowerBI |

### Visualizaciones (PNG)
| Archivo | Descripción |
|---------|-------------|
| `viz_01_historico_forecast.png` | Series temporales por categoría |
| `viz_02_comparacion_forecast.png` | Forecast superpuestos (7 categorías) |
| `viz_03_total_90d_barras.png` | Barras: total 90d por categoría |
| `viz_04_tendencia_semanal.png` | Tendencia semanal por categoría |
| `viz_05_incertidumbre.png` | Ranking de incertidumbre (ancho IC) |
| `viz_06_consolidado_diario.png` | Consolidado diario + banda IC |
| `viz_07_forecast_dia_semana.png` | Patrón por día de semana |
| `viz_08_distribucion_forecast.png` | Boxplot: distribución diaria |

### Reporte Cliente (PNG)
| Archivo | Descripción |
|---------|-------------|
| `client_01_comparacion_estrategias.png` | Compra tradicional vs forecast |
| `client_02_plan_compra_recomendado.png` | Plan con margen de seguridad |
| `client_03_historico_forecast.png` | Evolución + pronóstico por categoría |
| `client_04_incertidumbre_forecast.png` | Análisis de incertidumbre |
| `client_05_consolidado_diario.png` | Forecast consolidado diario |
| `client_06_forecast_dia_semana.png` | Forecast por día de semana |
| `client_07_ranking_volumen_valor.png` | Ranking por volumen y valor |
| `client_08_matriz_riesgo.png` | Matriz de riesgo por categoría |

### Reportes
| Archivo | Descripción |
|---------|-------------|
| `5.ClientReport.html` | Reporte en HTML |
| `5.ClientReport.pdf` | Reporte en PDF |
| `Reporte_Final.pdf` | Reporte final consolidado |

---

## Uso en PowerBI

El archivo `powerbi_forecast_completo.csv` está diseñado para PowerBI:

| Columna | Descripción | Uso en PowerBI |
|---------|-------------|----------------|
| `fecha` | Fecha (orden ascendente) | Eje de tiempo |
| `categoria` | Categoría del producto | Leyenda / Filtro |
| `tipo` | `Historico` o `Forecast` | Segmentador / Página separada |
| `cantidad` | Ventas reales (Histórico) o demanda pronosticada (Forecast) | Valor / Línea |
| `ci_lower` | Límite inferior IC 95% (solo Forecast) | Banda de confianza |
| `ci_upper` | Límite superior IC 95% (solo Forecast) | Banda de confianza |

**Contenido:** 4,603 filas — 3,966 históricas + 637 forecast — 7 categorías.

---

## Cómo Reproducir

```bash
# 1. Limpieza de datos
jupyter notebook 2.PreCleaning.ipynb

# 2. Análisis exploratorio
jupyter notebook 3.EDA.ipynb

# 3. Entrenar modelos Prophet (7 categorías)
jupyter notebook 7.WeeklyForecast.ipynb

# 4. Generar visualizaciones
jupyter notebook 4.Visualizations.ipynb

# 5. Generar reporte cliente (incluye exportación PowerBI)
jupyter notebook 5.ClientReport.ipynb

# 6. Exportar a PDF (opcional)
python exportarPDF.py
```

**Orden de ejecución:** 2 → 3 → 7 → 4 → 5

---

## Confidencialidad

> Los datos utilizados en este proyecto han sido publicados bajo autorización expresa de los clientes. Mantener la confidencialidad y anonimato de estos datos es de suma y extrema importancia.

---

---

# Logistics Forecasting with Prophet

90-day demand prediction for 7 food product categories, optimizing inventory and avoiding overstock for a food distribution company.

---

## Business Problem

The company purchased inventory without quantitative forecasting. In month 1 of 2023, inventory (2,002 units) exceeded actual sales (756 units) by **165%** — 1,246 units of immobilized capital. Since products are perishable (no returns for expired goods), overstock represents direct loss.

**Objective:** Predict daily demand by category over a 90-day horizon, enabling data-driven purchases instead of historical averages.

---

## Pipeline

```
1.BusinessAnalysis.ipynb   → Problem definition and scope
2.PreCleaning.ipynb        → Data cleaning and validation
3.EDA.ipynb                → Exploratory data analysis
7.WeeklyForecast.ipynb     → Prophet model training
4.Visualizations.ipynb     → General visualizations (8 charts)
5.ClientReport.ipynb       → Client-facing report
```

---

## Dataset

| Item | Value |
|------|-------|
| Raw records | 28,358 |
| Clean records | 22,640 |
| Historical period | 2023-01-03 → 2024-12-11 (603 days) |
| Products | 10 |
| Categories | 7 (Beverages, Meats, Frozen, Disposables, Sausages, Canned, Dairy) |
| Outliers detected (IQR) | 799 — kept (legitimate holiday spikes) |

**Data files:**
- `ventas_empresa_logistica.csv` — Raw source data
- `clean_data.csv` — Cleaned data (22,640 records, 10 columns)

---

## Methodology

**Model:** Prophet (Meta/Facebook) — selected over ARIMA for:
- Robust weekly seasonality handling
- Outlier and missing data tolerance
- Non-negative Bayesian confidence intervals (MCMC)
- Adaptive changepoint detection

**Training:**
- Weekly aggregation (W-MON) to reduce noise
- Daily expansion using historical day-of-week profiles
- Grid search: changepoint_prior_scale ∈ [0.001, 0.01, 0.05, 0.1] × yearly_seasonality ∈ [False, 3]
- 2 rolling window cross-validation
- Evaluation metric: SMAPE

**Results — Prophet vs ARIMA (SMAPE):**

| Category | ARIMA | Prophet v3 | Improvement |
|----------|-------|------------|-------------|
| Beverages | 48.7% | 46.6% | +2.1% |
| Meats | 41.3% | 35.9% | +5.4% |
| Frozen | 57.5% | 43.4% | +14.1% |
| Disposables | 43.9% | 32.3% | +11.6% |
| Sausages | 68.6% | 35.7% | +32.9% |
| Canned | 54.1% | 45.8% | +8.3% |
| Dairy | 42.5% | 32.3% | +10.2% |
| **Average** | **50.9%** | **38.9%** | **+12.1%** |

---

## Forecast Results (90 days)

| Category | Forecast Demand | Recommendation | Confidence |
|----------|----------------|---------------|------------|
| Disposables | 4,814 units | Standard purchase | High |
| Beverages | 2,174 units | Standard purchase | High |
| Dairy | 2,119 units | Standard purchase | High |
| Meats | 1,333 units | Standard purchase | High |
| Canned | 1,305 units | Standard purchase | High |
| Frozen | 819 units | Standard purchase | High |
| Sausages | 328 units | Standard purchase | High |

**Avoidable overstock savings:**
- Traditional purchase estimate: 12,328 units
- Forecast-based purchase: 12,891 units (with safety margin)
- Avoidable overstock: ~230 units (1.9%)
- **Warning:** Wide confidence intervals — 20-30% safety margin recommended

---

## PowerBI Usage

The `powerbi_forecast_completo.csv` file is designed for PowerBI:

| Column | Description | PowerBI Use |
|--------|-------------|-------------|
| `fecha` | Date (ascending order) | Time axis |
| `categoria` | Product category | Legend / Filter |
| `tipo` | `Historico` or `Forecast` | Slicer / Separate page |
| `cantidad` | Actual sales (Histórico) or forecast demand (Forecast) | Value / Line |
| `ci_lower` | 95% CI lower bound (Forecast only) | Confidence band |
| `ci_upper` | 95% CI upper bound (Forecast only) | Confidence band |

**Contents:** 4,603 rows — 3,966 historical + 637 forecast — 7 categories.

---

## Confidentiality

> The data used in this project has been published with the express authorization of the clients. Maintaining the confidentiality and anonymity of this data is of paramount and extreme importance.
