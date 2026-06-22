import os
from pathlib import Path
import subprocess
import tempfile
def convert_notebook_to_pdf(ipynb_path: str, output_pdf_path: str):
    """
    Convierte un notebook .ipynb a PDF usando nbconvert y playwright.

    El archivo intermedio HTML se maneja en memoria o mediante 
    un directorio temporal que es borrado automáticamente (tempfile).

    Args:
        ipynb_path: Ruta al archivo del notebook (.ipynb).
        output_pdf_path: Ruta donde se guardará el PDF final.
    """
    print(f"--- Iniciando conversión de {ipynb_path} a PDF ---")

    # 1. Validación y Preparación de Rutas
    if not Path(ipynb_path).exists():
        raise FileNotFoundError(f"Error: El archivo no fue encontrado en la ruta especificada: {ipynb_path}")

    # Usamos tempfile para crear un directorio temporal seguro 
    # donde salvaremos el HTML intermedio, garantizando que se borre.
    with tempfile.TemporaryDirectory() as tmpdir:
        html_intermediate_path = Path(tmpdir) / "temp_report.html"

        try:
            print("1/3: Conversión de .ipynb a HTML (usando nbconvert)...")
            
            # Ejecutar el comando nbconvert para generar el HTML
            result = subprocess.run(
                ['jupyter', 'nbconvert', '--to', 'html', ipynb_path, '--output', str(html_intermediate_path)],
                capture_output=True, 
                text=True,
                check=True  # Esto lanza un error si el comando falla
            )

            print("2/3: Cargando HTML en el navegador (playwright)...")
            
            # 3. Generación final del PDF con playwright
            print("3/3: Creando el archivo PDF y guardándolo...")
            
            # Importar playwright aquí por si hay algún problema global
            from playwright.sync_api import sync_playwright
            
            html_url = "file:///" + str(html_intermediate_path).replace("\\", "/")
            
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(html_url, wait_until="networkidle")
                page.pdf(path=output_pdf_path, format="A4", print_background=True)
                browser.close()

            print("\nExito: ¡Conversión completada exitosamente!")
            print(f"Archivo guardado en: {os.path.abspath(output_pdf_path)}")

        except subprocess.CalledProcessError as e:
            print(f"\nERROR durante la conversión de nbconvert.")
            print("Detalles del error (stdout):", e.stdout)
            print("Detalles del error (stderr):", e.stderr)
            return False
        except Exception as e:
            print(f"\nOcurrió un error inesperado durante el proceso: {e}")
            return False
    
    return True

# =========================
# USO DEL SCRIPT PRINCIPAL
# =========================

# Nombre de tu notebook
INPUT_NOTEBOOK = "5.ClientReport.ipynb" 
OUTPUT_PDF = "Reporte_Final.pdf"

try:
    convert_notebook_to_pdf(INPUT_NOTEBOOK, OUTPUT_PDF)
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: {e}")

