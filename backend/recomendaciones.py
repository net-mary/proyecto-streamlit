import logging
from typing import Dict, List, Optional
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generar_recomendaciones(emociones: List[Dict], audio: Dict, diagnostico: Optional[str] = None) -> List[str]:
    """
    Genera recomendaciones personalizadas basadas en análisis emocional y de audio.
    
    Args:
        emociones (List[Dict]): Resultados del análisis emocional
        audio (Dict): Resultados del análisis de audio
        diagnostico (str): Diagnóstico del niño (opcional)
        
    Returns:
        List[str]: Lista de recomendaciones personalizadas
    """
    recomendaciones = []
    
    try:
        # Análisis de contexto emocional
        contexto_emocional = _analizar_contexto_emocional(emociones)
        contexto_comunicativo = _analizar_contexto_comunicativo(audio)
        
        # Recomendaciones basadas en diagnóstico
        if diagnostico:
            recomendaciones.extend(_generar_recomendaciones_diagnostico(diagnostico, contexto_emocional, contexto_comunicativo))
        
        # Recomendaciones basadas en emociones
        recomendaciones.extend(_generar_recomendaciones_emocionales(contexto_emocional))
        
        # Recomendaciones basadas en comunicación
        recomendaciones.extend(_generar_recomendaciones_comunicativas(contexto_comunicativo))
        
        # Recomendaciones integradas
        recomendaciones.extend(_generar_recomendaciones_integradas(contexto_emocional, contexto_comunicativo, diagnostico))
        
        # Filtrar duplicados manteniendo orden
        recomendaciones_unicas = []
        for rec in recomendaciones:
            if rec not in recomendaciones_unicas:
                recomendaciones_unicas.append(rec)
        
        # Si no hay recomendaciones específicas, agregar por defecto
        if not recomendaciones_unicas:
            recomendaciones_unicas = _generar_recomendaciones_por_defecto()
        
        logger.info(f"Generadas {len(recomendaciones_unicas)} recomendaciones personalizadas")
        return recomendaciones_unicas
        
    except Exception as e:
        logger.error(f"Error generando recomendaciones: {e}")
        return _generar_recomendaciones_por_defecto()

def _analizar_contexto_emocional(emociones: List[Dict]) -> Dict:
    """
    Analiza el contexto emocional del niño basado en los resultados.
    
    Args:
        emociones (List[Dict]): Resultados emocionales
        
    Returns:
        Dict: Contexto emocional analizado
    """
    if not emociones:
        return {"patron": "sin_datos", "emociones_detectadas": 0}
    
    # Contar emociones y calcular métricas
    conteo_emociones = {}
    confianzas = []
    frames_con_emociones = 0
    
    for frame_result in emociones:
        emociones_frame = frame_result.get('emociones', [])
        if emociones_frame:
            frames_con_emociones += 1
            
        for emocion_data in emociones_frame:
            emocion = emocion_data.get('emotion', 'Unknown')
            confianza = emocion_data.get('confidence', 0.0)
            
            conteo_emociones[emocion] = conteo_emociones.get(emocion, 0) + 1
            confianzas.append(confianza)
    
    total_detecciones = sum(conteo_emociones.values())
    
    # Determinar emoción predominante
    emocion_predominante = max(conteo_emociones, key=conteo_emociones.get) if conteo_emociones else "Unknown"
    porcentaje_predominante = (conteo_emociones.get(emocion_predominante, 0) / total_detecciones * 100) if total_detecciones > 0 else 0
    
    # Calcular estabilidad emocional
    estabilidad = "alta" if porcentaje_predominante > 60 else "media" if porcentaje_predominante > 40 else "baja"
    
    # Clasificar emociones
    emociones_positivas = ["Happy", "Surprise"]
    emociones_negativas = ["Sad", "Angry", "Fear", "Disgust"]
    emociones_neutras = ["Neutral"]
    
    positivas_count = sum(conteo_emociones.get(emo, 0) for emo in emociones_positivas)
    negativas_count = sum(conteo_emociones.get(emo, 0) for emo in emociones_negativas)
    neutras_count = sum(conteo_emociones.get(emo, 0) for emo in emociones_neutras)
    
    # Determinar patrón emocional general
    if total_detecciones == 0:
        patron = "sin_detecciones"
    elif negativas_count > positivas_count * 1.5:
        patron = "predominio_negativo"
    elif positivas_count > negativas_count * 1.5:
        patron = "predominio_positivo"
    elif neutras_count > (positivas_count + negativas_count):
        patron = "predominio_neutral"
    else:
        patron = "equilibrado"
    
    # Evaluar variabilidad emocional
    variabilidad = len(conteo_emociones)
    if variabilidad <= 2:
        tipo_variabilidad = "baja"
    elif variabilidad <= 4:
        tipo_variabilidad = "media"
    else:
        tipo_variabilidad = "alta"
    
    return {
        "patron": patron,
        "emocion_predominante": emocion_predominante,
        "porcentaje_predominante": porcentaje_predominante,
        "estabilidad": estabilidad,
        "emociones_detectadas": total_detecciones,
        "variabilidad": tipo_variabilidad,
        "distribucion": conteo_emociones,
        "confianza_promedio": sum(confianzas) / len(confianzas) if confianzas else 0,
        "emociones_positivas": positivas_count,
        "emociones_negativas": negativas_count,
        "emociones_neutras": neutras_count,
        "cobertura_frames": (frames_con_emociones / len(emociones) * 100) if emociones else 0
    }

def _analizar_contexto_comunicativo(audio: Dict) -> Dict:
    """
    Analiza el contexto comunicativo basado en los resultados de audio.
    
    Args:
        audio (Dict): Resultados del análisis de audio
        
    Returns:
        Dict: Contexto comunicativo analizado
    """
    if not audio or audio.get('error'):
        return {"nivel": "sin_datos", "calidad": "no_evaluado"}
    
    # Extraer métricas básicas
    intentos = audio.get('intentos_comunicacion', 0)
    palabras_totales = audio.get('palabras_totales', 0)
    calidad = audio.get('calidad_comunicacion', 'no_evaluado')
    transcripcion = audio.get('transcription', '')
    palabras_infantiles = audio.get('palabras_infantiles', [])
    
    # Determinar nivel comunicativo
    if intentos == 0 and palabras_totales == 0:
        nivel = "no_verbal"
    elif intentos < 3:
        nivel = "pre_verbal"
    elif intentos < 8:
        nivel = "verbal_emergente"
    else:
        nivel = "verbal_funcional"
    
    # Evaluar claridad
    if not transcripcion:
        claridad = "inaudible"
    elif len(transcripcion) < 10:
        claridad = "muy_limitada"
    elif len(transcripcion) < 50:
        claridad = "limitada"
    else:
        claridad = "clara"
    
    # Evaluar complejidad del lenguaje
    if palabras_totales == 0:
        complejidad = "sin_lenguaje"
    elif palabras_totales < 5:
        complejidad = "palabras_simples"
    elif palabras_totales < 15:
        complejidad = "frases_basicas"
    else:
        complejidad = "lenguaje_elaborado"
    
    # Evaluar apropiación del vocabulario
    apropiacion_infantil = len(palabras_infantiles) / max(palabras_totales, 1) if palabras_totales > 0 else 0
    
    return {
        "nivel": nivel,
        "calidad": calidad,
        "claridad": claridad,
        "complejidad": complejidad,
        "intentos_comunicativos": intentos,
        "palabras_totales": palabras_totales,
        "palabras_infantiles_count": len(palabras_infantiles),
        "apropiacion_infantil": apropiacion_infantil,
        "longitud_transcripcion": len(transcripcion),
        "tiene_verbalizacion": bool(transcripcion)
    }

def _generar_recomendaciones_diagnostico(diagnostico: str, contexto_emocional: Dict, contexto_comunicativo: Dict) -> List[str]:
    """
    Genera recomendaciones específicas basadas en el diagnóstico.
    
    Args:
        diagnostico (str): Diagnóstico del niño
        contexto_emocional (Dict): Contexto emocional
        contexto_comunicativo (Dict): Contexto comunicativo
        
    Returns:
        List[str]: Recomendaciones específicas por diagnóstico
    """
    recomendaciones = []
    diagnostico_lower = diagnostico.lower()
    
    # AUTISMO / TEA
    if any(term in diagnostico_lower for term in ['autismo', 'tea', 'espectro']):
        recomendaciones.extend([
            "🔄 Implementar rutinas estructuradas y predecibles con apoyos visuales",
            "🎯 Usar sistemas de comunicación por intercambio de imágenes (PECS) si la comunicación verbal es limitada",
            "🌈 Crear un entorno sensorial controlado, evitando sobreestimulación"
        ])
        
        # Específicas por patrón emocional
        if contexto_emocional["patron"] == "predominio_negativo":
            recomendaciones.append("⚠️ Monitorear desregulación emocional; implementar estrategias de autorregulación específicas para TEA")
        
        # Específicas por comunicación
        if contexto_comunicativo["nivel"] == "no_verbal":
            recomendaciones.append("📱 Evaluar urgentemente sistemas de comunicación aumentativa y alternativa (CAA)")
        elif contexto_comunicativo["nivel"] == "verbal_emergente":
            recomendaciones.append("🗣️ Fomentar ecolalia funcional y expansión de vocabulario temático")
    
    # TDAH
    elif any(term in diagnostico_lower for term in ['tdah', 'atencion', 'hiperactividad', 'deficit']):
        recomendaciones.extend([
            "⏰ Dividir actividades en segmentos de 10-15 minutos con descansos activos",
            "🎯 Usar recordatorios visuales y auditivos para transiciones",
            "🏃 Incorporar movimiento físico en las actividades de aprendizaje"
        ])
        
        if contexto_emocional.get("variabilidad") == "alta":
            recomendaciones.append("📊 La alta variabilidad emocional puede indicar desregulación típica del TDAH; considerar técnicas de mindfulness adaptadas")
        
        if contexto_comunicativo["intentos_comunicativos"] < 5:
            recomendaciones.append("💬 La comunicación limitada puede estar relacionada con impulsividad; trabajar técnicas de pausa y reflexión")
    
    # SÍNDROME DE DOWN
    elif any(term in diagnostico_lower for term in ['down', 'trisomia']):
        recomendaciones.extend([
            "👁️ Priorizar aprendizaje visual sobre auditivo en todas las intervenciones",
            "🔁 Implementar repetición estructurada con refuerzo positivo inmediato",
            "👥 Fomentar interacciones sociales para desarrollo de habilidades comunicativas"
        ])
        
        if contexto_comunicativo["claridad"] == "muy_limitada":
            recomendaciones.append("👄 Considerar terapia orofacial para mejorar articulación y claridad del habla")
    
    # PARÁLISIS CEREBRAL
    elif any(term in diagnostico_lower for term in ['paralisis', 'cerebral', 'pc']):
        recomendaciones.extend([
            "🔧 Implementar adaptaciones físicas y tecnológicas según capacidades motoras",
            "📱 Evaluar dispositivos de comunicación asistiva si hay limitaciones del habla",
            "🤝 Coordinar con terapia ocupacional y fisioterapia para enfoque integral"
        ])
        
        if contexto_comunicativo["nivel"] == "no_verbal":
            recomendaciones.append("🖥️ Priorizar sistemas de comunicación por switch o mirada según capacidades motoras")
    
    # DISCAPACIDAD INTELECTUAL
    elif any(term in diagnostico_lower for term in ['intelectual', 'cognitiva', 'retraso']):
        recomendaciones.extend([
            "📚 Adaptar contenidos a nivel cognitivo con materiales concretos y visuales",
            "🎓 Dividir objetivos en pequeños pasos con celebración de logros",
            "👨‍👩‍👧 Involucrar activamente a la familia en estrategias de refuerzo"
        ])
        
        if contexto_emocional["patron"] == "predominio_negativo":
            recomendaciones.append("😊 La frustración puede estar relacionada con demandas cognitivas; ajustar expectativas y aumentar apoyo")
    
    # TRASTORNOS DEL LENGUAJE
    elif any(term in diagnostico_lower for term in ['lenguaje', 'habla', 'comunicacion']):
        recomendaciones.extend([
            "🗣️ Implementar terapia del lenguaje intensiva con enfoque funcional",
            "🎵 Usar técnicas de prosodia y ritmo para mejorar fluidez",
            "👂 Fomentar comprensión auditiva antes que expresión verbal"
        ])
        
        if contexto_comunicativo["complejidad"] == "sin_lenguaje":
            recomendaciones.append("🚨 Evaluación integral del lenguaje urgente; considerar trastornos asociados")
    
    return recomendaciones

def _generar_recomendaciones_emocionales(contexto_emocional: Dict) -> List[str]:
    """
    Genera recomendaciones basadas en el patrón emocional detectado.
    
    Args:
        contexto_emocional (Dict): Contexto emocional analizado
        
    Returns:
        List[str]: Recomendaciones emocionales
    """
    recomendaciones = []
    patron = contexto_emocional.get("patron", "sin_datos")
    emocion_predominante = contexto_emocional.get("emocion_predominante", "Unknown")
    
    # Recomendaciones por patrón general
    if patron == "predominio_negativo":
        recomendaciones.extend([
            "⚠️ Se detectó predominio de emociones negativas - evaluación psicoemocional recomendada",
            "🌟 Implementar actividades de regulación emocional y bienestar",
            "🎨 Fomentar expresión creativa (arte, música) para canalizar emociones",
            "💝 Aumentar refuerzos positivos y celebración de logros pequeños"
        ])
    
    elif patron == "predominio_positivo":
        recomendaciones.extend([
            "😊 Excelente regulación emocional detectada - mantener estrategias actuales",
            "📈 Aprovechar estado emocional positivo para nuevos aprendizajes",
            "🎯 Usar emociones positivas como refuerzo natural en actividades"
        ])
    
    elif patron == "predominio_neutral":
        recomendaciones.extend([
            "😐 Expresión emocional limitada - estimular variabilidad expresiva",
            "🎭 Implementar juegos de expresión facial y reconocimiento emocional",
            "📚 Usar cuentos e historias sociales para enseñar emociones"
        ])
    
    # Recomendaciones por emoción específica predominante
    if emocion_predominante == "Sad":
        porcentaje = contexto_emocional.get("porcentaje_predominante", 0)
        if porcentaje > 50:
            recomendaciones.append("😢 Alta frecuencia de tristeza detectada - considerar evaluación de depresión infantil")
        recomendaciones.extend([
            "🎵 Implementar musicoterapia y actividades que generen bienestar",
            "🤗 Aumentar tiempo de interacción social positiva y juego colaborativo",
            "🏃 Incluir actividad física regular para mejorar estado de ánimo"
        ])
    
    elif emocion_predominante == "Angry":
        recomendaciones.extend([
            "😤 Enseñar técnicas de autorregulación apropiadas para la edad",
            "🧘 Implementar técnicas de relajación y mindfulness infantil",
            "📖 Usar historias sociales sobre manejo de la frustración",
            "🎯 Identificar y modificar disparadores de enojo"
        ])
    
    elif emocion_predominante == "Fear":
        recomendaciones.extend([
            "😰 Trabajar técnicas de desensibilización gradual para miedos",
            "🛡️ Crear entorno seguro y predecible para reducir ansiedad",
            "🎮 Usar juego terapéutico para procesar temores",
            "👨‍👩‍👧 Involucrar a cuidadores en estrategias de manejo de ansiedad"
        ])
    
    elif emocion_predominante == "Happy":
        recomendaciones.extend([
            "🎉 Estado emocional positivo detectado - excelente base para aprendizaje",
            "📚 Aprovechar motivación alta para introducir nuevas habilidades",
            "🎯 Usar refuerzo positivo natural ya presente"
        ])
    
    # Recomendaciones por estabilidad emocional
    estabilidad = contexto_emocional.get("estabilidad", "media")
    if estabilidad == "baja":
        recomendaciones.append("🌊 Variabilidad emocional alta detectada - trabajar estrategias de estabilización")
    elif estabilidad == "alta":
        if contexto_emocional.get("variabilidad") == "baja":
            recomendaciones.append("📊 Expresión emocional muy limitada - estimular rango expresivo")
    
    return recomendaciones

def _generar_recomendaciones_comunicativas(contexto_comunicativo: Dict) -> List[str]:
    """
    Genera recomendaciones basadas en el análisis comunicativo.
    
    Args:
        contexto_comunicativo (Dict): Contexto comunicativo
        
    Returns:
        List[str]: Recomendaciones comunicativas
    """
    recomendaciones = []
    nivel = contexto_comunicativo.get("nivel", "sin_datos")
    claridad = contexto_comunicativo.get("claridad", "no_evaluado")
    
    # Recomendaciones por nivel comunicativo
    if nivel == "no_verbal":
        recomendaciones.extend([
            "🚨 Ausencia de comunicación verbal - evaluación urgente de CAA (Comunicación Aumentativa y Alternativa)",
            "👋 Fomentar comunicación gestual y señalamiento funcional",
            "📱 Considerar aplicaciones de comunicación por imágenes (PECS digital)",
            "🎯 Establecer intención comunicativa antes que forma verbal"
        ])
    
    elif nivel == "pre_verbal":
        recomendaciones.extend([
            "🗣️ Estimular vocalización mediante imitación y juego vocal",
            "🎵 Usar técnicas de comunicación total (gesto + verbalización)",
            "📖 Implementar rutinas de lectura interactiva diaria",
            "👄 Considerar estimulación orofacial si hay dificultades articulatorias"
        ])
    
    elif nivel == "verbal_emergente":
        recomendaciones.extend([
            "📈 Expandir vocabulario funcional mediante rutinas diarias",
            "🔄 Usar técnicas de modelado y expansión de frases",
            "🎭 Implementar juegos de imitación vocal y verbal",
            "📚 Crear oportunidades de comunicación espontánea"
        ])
    
    elif nivel == "verbal_funcional":
        recomendaciones.extend([
            "💬 Fomentar conversación elaborada y narrativa",
            "📖 Trabajar comprensión de textos y seguimiento de instrucciones complejas",
            "🎯 Desarrollar habilidades pragmáticas del lenguaje"
        ])
    
    # Recomendaciones por claridad
    if claridad == "inaudible" or claridad == "muy_limitada":
        recomendaciones.extend([
            "👂 Evaluación audiológica para descartar pérdida auditiva",
            "🔊 Trabajar proyección de voz y articulación",
            "🎤 Considerar amplificación o sistemas FM si es necesario"
        ])
    
    elif claridad == "limitada":
        recomendaciones.append("🗣️ Terapia del habla enfocada en inteligibilidad")
    
    # Recomendaciones por complejidad
    complejidad = contexto_comunicativo.get("complejidad", "sin_lenguaje")
    if complejidad == "palabras_simples":
        recomendaciones.append("🔗 Fomentar combinación de palabras en frases simples")
    elif complejidad == "frases_basicas":
        recomendaciones.append("📝 Trabajar estructura gramatical básica y ampliación de frases")
    
    # Apropiación del vocabulario infantil
    apropiacion = contexto_comunicativo.get("apropiacion_infantil", 0)
    if apropiacion < 0.3 and contexto_comunicativo.get("palabras_totales", 0) > 5:
        recomendaciones.append("👶 Fomentar vocabulario apropiado para la edad cronológica")
    
    return recomendaciones

def _generar_recomendaciones_integradas(contexto_emocional: Dict, contexto_comunicativo: Dict, diagnostico: str = None) -> List[str]:
    """
    Genera recomendaciones que integran aspectos emocionales y comunicativos.
    
    Args:
        contexto_emocional (Dict): Contexto emocional
        contexto_comunicativo (Dict): Contexto comunicativo  
        diagnostico (str): Diagnóstico si está disponible
        
    Returns:
        List[str]: Recomendaciones integradas
    """
    recomendaciones = []
    
    # Integración emoción-comunicación
    patron_emocional = contexto_emocional.get("patron", "")
    nivel_comunicativo = contexto_comunicativo.get("nivel", "")
    
    # Frustración por limitaciones comunicativas
    if (patron_emocional == "predominio_negativo" and 
        contexto_emocional.get("emocion_predominante") in ["Angry", "Sad"] and
        nivel_comunicativo in ["no_verbal", "pre_verbal"]):
        
        recomendaciones.append("🔄 La frustración emocional puede estar relacionada con limitaciones comunicativas - priorizar desarrollo de comunicación funcional")
    
    # Comunicación limitada con patrón neutral
    if (patron_emocional == "predominio_neutral" and 
        nivel_comunicativo in ["no_verbal", "pre_verbal"]):
        
        recomendaciones.append("📈 Combinar estimulación emocional y comunicativa mediante juego interactivo estructurado")
    
    # Alta variabilidad emocional con comunicación funcional
    if (contexto_emocional.get("variabilidad") == "alta" and 
        nivel_comunicativo == "verbal_funcional"):
        
        recomendaciones.append("🗣️ Usar habilidades verbales para enseñar autorregulación emocional")
    
    # Recomendaciones por confianza en detecciones
    confianza_promedio = contexto_emocional.get("confianza_promedio", 0)
    if confianza_promedio < 0.5:
        recomendaciones.append("📸 Baja confianza en detección emocional - considerar mejores condiciones de grabación para futuros análisis")
    
    # Cobertura de frames baja
    cobertura = contexto_emocional.get("cobertura_frames", 0)
    if cobertura < 50:
        recomendaciones.append("🎥 Baja detección facial - asegurar buena iluminación y posición del niño frente a la cámara")
    
    # Recomendaciones de seguimiento
    recomendaciones.extend([
        "📅 Realizar seguimiento en 2-3 semanas para evaluar progreso",
        "👨‍👩‍👧‍👦 Involucrar a todos los cuidadores en la implementación de estrategias",
        "📊 Documentar cambios observados para ajustar intervenciones"
    ])
    
    return recomendaciones

def _generar_recomendaciones_por_defecto() -> List[str]:
    """
    Genera recomendaciones por defecto cuando no se pueden generar específicas.
    
    Returns:
        List[str]: Recomendaciones por defecto
    """
    return [
        "🔍 Realizar observación sistemática del comportamiento en diferentes contextos",
        "📝 Mantener registro diario de comunicación y expresiones emocionales",
        "👨‍⚕️ Consultar con equipo interdisciplinario para evaluación completa",
        "🏠 Crear ambiente estructurado y predecible en el hogar",
        "💪 Reforzar fortalezas observadas mientras se trabajan áreas de mejora",
        "📈 Establecer objetivos realistas y medibles a corto plazo",
        "🤝 Mantener comunicación constante entre familia y profesionales"
    ]

def generar_reporte_recomendaciones(recomendaciones: List[str], contexto_emocional: Dict, 
                                   contexto_comunicativo: Dict, diagnostico: str = None) -> Dict:
    """
    Genera reporte estructurado de recomendaciones con contexto detallado.
    
    Args:
        recomendaciones (List[str]): Lista de recomendaciones
        contexto_emocional (Dict): Contexto emocional
        contexto_comunicativo (Dict): Contexto comunicativo
        diagnostico (str): Diagnóstico del niño
        
    Returns:
        Dict: Reporte estructurado
    """
    # Categorizar recomendaciones por tipo
    categorias = {
        "urgentes": [],
        "emocionales": [],
        "comunicativas": [],
        "familiares": [],
        "profesionales": [],
        "seguimiento": []
    }
    
    for rec in recomendaciones:
        if any(palabra in rec.lower() for palabra in ['urgente', '🚨', 'inmediato', 'evaluar urgentemente']):
            categorias["urgentes"].append(rec)
        elif any(palabra in rec.lower() for palabra in ['emocional', '😢', '😤', '😰', '🧘', 'regulación']):
            categorias["emocionales"].append(rec)
        elif any(palabra in rec.lower() for palabra in ['comunicación', '🗣️', '📱', 'verbal', 'lenguaje', 'caa']):
            categorias["comunicativas"].append(rec)
        elif any(palabra in rec.lower() for palabra in ['familia', '👨‍👩‍👧', 'cuidadores', 'hogar']):
            categorias["familiares"].append(rec)
        elif any(palabra in rec.lower() for palabra in ['profesional', '👨‍⚕️', 'terapia', 'evaluación']):
            categorias["profesionales"].append(rec)
        elif any(palabra in rec.lower() for palabra in ['seguimiento', '📅', 'documentar', 'progreso']):
            categorias["seguimiento"].append(rec)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "diagnostico": diagnostico or "No especificado",
        "resumen_emocional": contexto_emocional,
        "resumen_comunicativo": contexto_comunicativo,
        "recomendaciones_por_categoria": categorias,
        "total_recomendaciones": len(recomendaciones),
        "nivel_prioridad": "alta" if categorias["urgentes"] else "media" if categorias["profesionales"] else "normal"
    }

def validar_recomendaciones(recomendaciones: List[str]) -> Dict[str, bool]:
    """
    Valida la calidad y completitud de las recomendaciones generadas.
    
    Args:
        recomendaciones (List[str]): Lista de recomendaciones
        
    Returns:
        Dict[str, bool]: Resultados de validación
    """
    validacion = {
        "tiene_recomendaciones": len(recomendaciones) > 0,
        "longitud_adecuada": 3 <= len(recomendaciones) <= 15,
        "incluye_emocionales": any("emocional" in rec.lower() or any(emoji in rec for emoji in ['😢', '😤', '😰', '🧘']) for rec in recomendaciones),
        "incluye_comunicativas": any("comunicación" in rec.lower() or "verbal" in rec.lower() or "🗣️" in rec for rec in recomendaciones),
        "incluye_seguimiento": any("seguimiento" in rec.lower() or "📅" in rec for rec in recomendaciones),
        "sin_duplicados": len(recomendaciones) == len(set(recomendaciones))
    }
    
    validacion["es_completa"] = all([
        validacion["tiene_recomendaciones"],
        validacion["longitud_adecuada"],
        validacion["sin_duplicados"]
    ])
    
    return validacion

