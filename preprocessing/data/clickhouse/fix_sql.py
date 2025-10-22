

if __name__ == "__main__":
    # Params files
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    import sys
import json
from pathlib import Path

def unescape_file(input_path, output_path=None):
    """
    Entschlüsselt escapte Zeichen in einer Datei und speichert das Ergebnis.
    Args:
        input_path (str): Pfad zur Eingabedatei.
        output_path (str, optional): Pfad zur Ausgabedatei. Wenn None, wird die Originaldatei überschrieben.
    """
    try:
        # Prüfe, ob die Datei existiert
        if not Path(input_path).is_file():
            print(f"Fehler: {input_path} existiert nicht oder ist keine Datei.")
            return

        # Datei einlesen
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Versuche zuerst JSON
        try:
            data = json.loads(content)
            decoded_content = json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # Fallback auf unicode_escape für einfache Textdateien
            decoded_content = bytes(content, "utf-8").decode("unicode_escape")

        # Ausgabepfad festlegen (Original überschreiben oder neue Datei)
        if output_path is None:
            output_path = input_path
        else:
            # Stelle sicher, dass das Ausgabeverzeichnis existiert
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Ergebnis speichern
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(decoded_content)
        print(f"Erfolgreich verarbeitet: {input_path} -> {output_path}")

    except UnicodeDecodeError as e:
        print(f"Fehler bei {input_path}: Ungültige Escape-Sequenz oder Encoding ({e})")
    except Exception as e:
        print(f"Fehler bei {input_path}: {e}")

if __name__ == "__main__":
    # Prüfe, ob Dateipfade als Argumente übergeben wurden
    if len(sys.argv) < 2:
        print("Verwendung: python unescape_files.py <datei1> <datei2> ...")
        sys.exit(1)

    # Liste der Dateipfade aus Kommandozeilenargumenten (überspringt argv[0], das der Skriptname ist)
    files_to_process = sys.argv[1:]

    # Optional: Ausgabeverzeichnis für verarbeitete Dateien (None, um Originale zu überschreiben)
    output_directory = None  # Ersetze mit "pfad/zu/deinem/ausgabe_verzeichnis" für neue Dateien

    for file_path in files_to_process:
        if output_directory:
            # Erstelle Ausgabepfad mit gleichem Dateinamen im Ausgabeverzeichnis
            output_path = str(Path(output_directory) / Path(file_path).name)
        else:
            output_path = None  # Originaldatei überschreiben

        unescape_file(file_path, output_path)