# 🌍 Dashboard — Cambio Climático: CO₂, Combustibles Fósiles y Temperatura

Dashboard interactivo del Proyecto Integrador de Ciencia de Datos Ambientales.
Analiza la relación entre las emisiones de CO₂, el consumo de combustibles fósiles
y la temperatura media global (10 países, 1970–2024).

## Secciones
- **Inicio** — resumen y serie global de temperatura.
- **Exploración (EDA)** — evolución por país, estadística descriptiva, boxplots, datos crudos.
- **Relación global** — regresión temperatura ~ CO₂ / fósil, con R² y validación cruzada.
- **Relación por país** — correlaciones firmadas (el hallazgo de Europa con signo negativo).
- **Matriz predictiva** — temperatura 2050 según niveles de CO₂ × consumo fósil (4×4).
- **Proyección a 2050** — global (año → variable → temperatura) y por país, con backtest.
- **Comparación de modelos** — Regresión Lineal vs SVR por país (justifica la elección).
- **Exploración inicial (Kaggle)** — gráficos del dataset sintético usado en la fase de decisión.
- **Conclusiones**.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
Abre en el navegador `http://localhost:8501`.

## Desplegar en Streamlit Community Cloud (gratis, desde GitHub)

1. Crea un repositorio en GitHub y sube **estos archivos** juntos:
   - `app.py`
   - `requirements.txt`
   - `Base_de_Datos_Proyecto.csv`  ← base real, imprescindible.
   - `Kaggle_Climate_Dataset.csv`  ← dataset sintético (sección "Exploración inicial"; si falta, esa sección avisa pero el resto funciona).

   ```bash
   git init
   git add app.py requirements.txt Base_de_Datos_Proyecto.csv Kaggle_Climate_Dataset.csv README.md
   git commit -m "Dashboard clima"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main
   ```

2. Entra a **https://share.streamlit.io** e inicia sesión con GitHub.
3. Clic en **New app** → elige tu repositorio, la rama `main` y el archivo `app.py`.
4. **Deploy**. En 1–2 minutos tendrás una URL pública para compartir.

> Si cambias el nombre del CSV, actualiza también la función `cargar_datos()` en `app.py`.
