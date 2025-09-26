import os

def validar_video(video_file, max_size_bytes=200*1024*1024, valid_extensions=None):
    if valid_extensions is None:
        valid_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    nombre = video_file.name.lower()
    if not any(nombre.endswith(ext) for ext in valid_extensions):
        return False, f"Formato no soportado. Use {', '.join(valid_extensions)}"
    size = getattr(video_file, "size", None)
    if size is not None and size > max_size_bytes:
        return False, f"Archivo demasiado grande. Max {max_size_bytes/(1024*1024)} MB"
    return True, "Archivo válido"

