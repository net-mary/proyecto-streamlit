#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configurador de Modelos de IA
==============================

Script para descargar y configurar los modelos necesarios para el análisis emocional.

Uso:
    python setup_models.py
"""

import os
import sys

def verificar_modelos():
    """Verifica qué modelos están disponibles."""
    print("\n" + "="*60)
    print("VERIFICACIÓN DE MODELOS")
    print("="*60 + "\n")
    
    models_dir = "./models"
    
    if not os.path.exists(models_dir):
        print(f"⚠️  La carpeta {models_dir}/ no existe")
        print(f"   Creando carpeta...")
        os.makedirs(models_dir, exist_ok=True)
        print(f"   ✅ Carpeta creada\n")
    
    modelos_requeridos = {
        "FER_model.h5": "Modelo FER principal",
        "fer2013_mini_XCEPTION.99-0.65.hdf5": "Modelo mini-XCEPTION",
        "haarcascade_frontalface_default.xml": "Clasificador Haar Cascade"
    }
    
    modelos_encontrados = []
    modelos_faltantes = []
    
    for archivo, descripcion in modelos_requeridos.items():
        ruta = os.path.join(models_dir, archivo)
        if os.path.exists(ruta):
            tamaño = os.path.getsize(ruta) / (1024 * 1024)  # MB
            print(f"✅ {descripcion}")
            print(f"   Archivo: {archivo} ({tamaño:.2f} MB)")
            modelos_encontrados.append(archivo)
        else:
            print(f"❌ {descripcion}")
            print(f"   Archivo faltante: {archivo}")
            modelos_faltantes.append(archivo)
    
    print(f"\n📊 Resumen:")
    print(f"   Modelos encontrados: {len(modelos_encontrados)}/{len(modelos_requeridos)}")
    print(f"   Modelos faltantes: {len(modelos_faltantes)}")
    
    return modelos_faltantes

def descargar_haarcascade():
    """Descarga el clasificador Haar Cascade desde OpenCV."""
    print("\n" + "="*60)
    print("DESCARGANDO HAARCASCADE")
    print("="*60 + "\n")
    
    try:
        import cv2
        
        # Primero intentar copiar desde OpenCV instalado
        print("📥 Intentando copiar desde OpenCV instalado...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        destino = "./models/haarcascade_frontalface_default.xml"
        
        if os.path.exists(cascade_path):
            import shutil
            shutil.copy(cascade_path, destino)
            print(f"✅ Copiado desde OpenCV: {destino}")
            return True
        
        # Si no está en OpenCV, descargar de GitHub
        print("📥 Descargando desde repositorio de OpenCV...")
        import urllib.request
        
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        
        urllib.request.urlretrieve(url, destino)
        
        if os.path.exists(destino) and os.path.getsize(destino) > 1000:
            print(f"✅ Archivo descargado: {destino}")
            return True
        else:
            print("❌ Error: Archivo descargado inválido")
            return False
            
    except Exception as e:
        print(f"❌ Error descargando haarcascade: {e}")
        return False

def instrucciones_modelos_fer():
    """Muestra instrucciones para obtener los modelos FER."""
    print("\n" + "="*60)
    print("CÓMO OBTENER LOS MODELOS FER")
    print("="*60 + "\n")
    
    print("Los modelos de reconocimiento emocional FER son modelos pre-entrenados")
    print("que deben obtenerse de fuentes confiables.\n")
    
    print("📦 OPCIÓN 1: Modelos Pre-entrenados")
    print("-" * 40)
    print("1. Busca modelos FER2013 pre-entrenados en:")
    print("   • GitHub: https://github.com/oarriaga/face_classification")
    print("   • Kaggle: https://www.kaggle.com/datasets/msambare/fer2013")
    print()
    print("2. Descarga los siguientes archivos:")
    print("   • FER_model.h5")
    print("   • fer2013_mini_XCEPTION.99-0.65.hdf5")
    print()
    print("3. Colócalos en la carpeta: ./models/")
    print()
    
    print("📦 OPCIÓN 2: Entrenar tus propios modelos")
    print("-" * 40)
    print("Si prefieres entrenar tus propios modelos:")
    print("1. Descarga el dataset FER2013")
    print("2. Usa frameworks como TensorFlow/Keras")
    print("3. Entrena y guarda los modelos en ./models/")
    print()
    
    print("💡 OPCIÓN 3: Modo Fallback (Sin modelos)")
    print("-" * 40)
    print("El sistema puede funcionar SIN los modelos FER usando un modo fallback")
    print("que realiza análisis básico basado en reglas. Aunque menos preciso,")
    print("permite probar el sistema completo.")
    print()
    
    print("⚠️  IMPORTANTE:")
    print("   Los modelos FER son grandes (50-100 MB cada uno)")
    print("   Asegúrate de tener espacio suficiente en disco")

def crear_readme_models():
    """Crea un README en la carpeta models."""
    readme_content = """# Modelos de IA para Análisis Emocional

Esta carpeta debe contener los modelos pre-entrenados necesarios para el análisis emocional.

## Modelos Requeridos:

### 1. FER_model.h5
- **Descripción**: Modelo principal de reconocimiento emocional
- **Framework**: Keras/TensorFlow
- **Tamaño**: ~50 MB
- **Fuente**: Entrenado en dataset FER2013

### 2. fer2013_mini_XCEPTION.99-0.65.hdf5
- **Descripción**: Modelo mini-XCEPTION optimizado
- **Framework**: Keras/TensorFlow
- **Tamaño**: ~100 MB
- **Fuente**: Arquitectura Xception adaptada para FER2013

### 3. haarcascade_frontalface_default.xml
- **Descripción**: Clasificador Haar Cascade para detección facial
- **Framework**: OpenCV
- **Tamaño**: ~1 MB
- **Fuente**: OpenCV oficial

## Cómo obtener los modelos:

### Opción 1: Descargar Pre-entrenados
Busca modelos compatibles en:
- GitHub: https://github.com/oarriaga/face_classification
- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

### Opción 2: Modo Fallback
El sistema puede funcionar sin modelos usando análisis basado en reglas.
La precisión será menor pero permite probar el sistema.

## Verificación:
Ejecuta: `python setup_models.py` para verificar modelos instalados.
"""
    
    readme_path = "./models/README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ README creado en: {readme_path}")

def main():
    """Función principal."""
    print("\n" + "="*60)
    print("🤖 CONFIGURADOR DE MODELOS DE IA")
    print("Sistema de Análisis Emocional Multimodal")
    print("="*60)
    
    # Verificar modelos
    modelos_faltantes = verificar_modelos()
    
    # Descargar Haar Cascade si falta
    if "haarcascade_frontalface_default.xml" in modelos_faltantes:
        if descargar_haarcascade():
            modelos_faltantes.remove("haarcascade_frontalface_default.xml")
    
    # Mostrar instrucciones para modelos FER
    if any("FER" in m or "fer" in m for m in modelos_faltantes):
        instrucciones_modelos_fer()
    
    # Crear README
    crear_readme_models()
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60 + "\n")
    
    if not modelos_faltantes:
        print("🎉 ¡TODOS LOS MODELOS ESTÁN DISPONIBLES!")
        print("   El sistema está listo para funcionar con máxima precisión.")
    else:
        print("⚠️  MODELOS FALTANTES:")
        for modelo in modelos_faltantes:
            print(f"   • {modelo}")
        print()
        print("💡 El sistema puede funcionar en MODO FALLBACK")
        print("   pero la precisión será limitada.")
        print()
        print("   Revisa las instrucciones anteriores para obtener los modelos.")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()