"""Recherche de restaurants par diagramme de Voronoï et triangulation de Delaunay (VTK + PySide6).

Utilisation :
    python app.py [fichier.csv]     lance l'application (par défaut data/restaurants.csv)
    python app.py --test            lance les tests du moteur géométrique et des scènes 3D
    python app.py --capture img.png [--vue carte|cones|paraboloide] [--pos X Y] [--mode mosaique|cuisine|note]
                  [--delaunay] [--comparer]
                                    enregistre une capture de la fenêtre sans l'afficher
"""
import subprocess
import sys
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
CSV_DEFAUT = DOSSIER / "data" / "restaurants.csv"


def valeur(args, option, n=1):
    if option not in args:
        return None
    k = args.index(option)
    return args[k + 1:k + 1 + n] if n > 1 else args[k + 1]


def main(args):
    if "--test" in args:
        # Un processus par suite : chacune se termine par sys.exit, qui arrêterait les suivantes.
        codes = [subprocess.call([sys.executable, str(DOSSIER / "tests" / nom)])
                 for nom in ("test_geometrie.py", "test_scenes3d.py")]
        return max(codes)

    import vtk
    # VTK journalise des avertissements verbeux (validation du Voronoï) : on ne garde que les erreurs.
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    import geometrie
    import interface

    app = QApplication(sys.argv[:1])
    interface.appliquer_theme(app)
    capture = valeur(args, "--capture")
    chemin = next((Path(a) for a in args if a.lower().endswith(".csv") and a != capture), CSV_DEFAUT)
    try:
        restaurants = geometrie.charger_csv(chemin)
    except geometrie.ErreurDonnees as erreur:
        if capture:  # sans fenêtre : pas de boîte de dialogue qui attendrait un clic
            print(f"{chemin.name} : {erreur}", file=sys.stderr)
        else:
            QMessageBox.critical(None, "Fichier refusé", f"{chemin.name}\n\n{erreur}\n\n{interface.CONSEIL_CSV}")
        return 1

    # Le tout premier appel VTK charge ses bibliothèques (près de 100 ms) : on le fait ici pour que
    # les temps affichés dans la barre d'état mesurent le calcul et non ce chargement.
    geometrie.Carte(restaurants[:4])

    fenetre = interface.FenetrePrincipale(restaurants, source=chemin.name, hors_ecran=bool(capture))
    if not capture:
        fenetre.show()
        return app.exec()

    # Capture sans afficher la fenêtre : utile pour le rapport et pour vérifier l'interface.
    fenetre.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    fenetre.show()
    app.processEvents()
    if valeur(args, "--mode"):
        fenetre.explorer.segments_mode.choisir(valeur(args, "--mode"))
        fenetre._mode(valeur(args, "--mode"))
    if "--delaunay" in args:
        fenetre.explorer.options["delaunay"].setChecked(True)
    if valeur(args, "--pos", 2):
        x, y = (float(v) for v in valeur(args, "--pos", 2))
        fenetre.choisir(x, y)
    if valeur(args, "--vue"):
        fenetre.changer_vue(valeur(args, "--vue"))

    def finir():
        fenetre.capture_complete(capture)
        print(f"Capture enregistrée : {capture}")
        fenetre.close()  # libère VTK proprement avant de quitter
        app.quit()

    QTimer.singleShot(2300, finir)  # laisse finir les animations (caméra 3D, relèvement)
    app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
