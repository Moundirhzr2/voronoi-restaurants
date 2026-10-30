"""Extension D : temps de construction et de recherche du modèle Carte."""

import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from geometrie import Carte, charger_csv, generer_restaurants


SIZES = (30, 100, 1000, 10000, 100000)
COLUMNS = ("N", "construction_ms", "voronoi_ms", "delaunay_ms", "localisation_ms",
           "plus_proche_1000q_ms", "brute_1000q_ms")


def mesurer(operation):
    """Médiane de trois exécutions, sans compter la libération du résultat."""
    durees, resultat = [], None
    for _ in range(3):
        resultat = None
        debut = perf_counter()
        resultat = operation()
        durees.append((perf_counter() - debut) * 1000)
    return median(durees), resultat


def brute_numpy(sites, positions):
    """Recherche exhaustive, sans matrice de distances 1000 × N en mémoire."""
    indices = np.empty(len(positions), dtype=np.int64)
    for i, position in enumerate(positions):
        delta = sites - position
        indices[i] = np.einsum("ij,ij->i", delta, delta).argmin()
    return indices


def duree(ms):
    """Durée en français : 0,3 ms, 15 ms, 7,1 s."""
    texte = f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"
    return texte.replace(".", ",").replace(" ", " ")


def tracer(lignes, chemin):
    """Exporter les courbes log-log, aux couleurs sombres de l'application, si matplotlib est disponible."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    except ImportError:
        print("matplotlib absent : graphique ignoré.")
        return
    fond, encre, discret, trait = "#0f1011", "#d0d6e0", "#8a8f98", "#23252a"
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
                         "axes.labelcolor": discret, "xtick.color": discret, "ytick.color": discret})
    donnees = np.array(lignes)
    n = donnees[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=fond)
    # Palettes vérifiées avec le script du skill dataviz (fond sombre, daltonismes) ; marqueurs et
    # valeurs de fin servent d'encodage secondaire.
    panneaux = ((axes[0], "Construction de la carte", ((1, "Carte complète", "#3987e5", "o"),
                                                        (2, "Voronoï", "#c98500", "s"),
                                                        (3, "Delaunay", "#d55181", "^"),
                                                        (4, "Localisateur", "#008300", "D"))),
                (axes[1], "1 000 recherches du plus proche", ((5, "Plus proche (VTK)", "#3987e5", "o"),
                                                               (6, "Force brute (NumPy)", "#d95926", "s"))))
    for ax, titre, series in panneaux:
        fins = np.log10([donnees[-1, colonne] for colonne, *_ in series])
        places = fins.copy()
        for a, b in zip(np.argsort(fins), np.argsort(fins)[1:]):  # écarter les étiquettes trop proches
            places[b] = max(places[b], places[a] + 0.28)
        for (colonne, legende, couleur, marque), place in zip(series, places):
            ax.plot(n, donnees[:, colonne], color=couleur, marker=marque, linewidth=2, markersize=6,
                    markeredgecolor=fond, markeredgewidth=1.5, label=legende)
            ax.annotate(duree(donnees[-1, colonne]), (n[-1] * 1.35, 10 ** place), va="center",
                        color=encre, fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor(fond)
        ax.set_title(titre, loc="left", color="#f7f8f8", fontsize=13, pad=14)
        ax.set_xlabel("Nombre de restaurants", labelpad=8)
        ax.set_xlim(n[0] / 1.5, n[-1] * 4.5)
        valeurs = donnees[:, [colonne for colonne, *_ in series]]
        ax.set_ylim(valeurs.min() / 1.6, valeurs.max() * 2.2)  # au moins une graduation sous chaque courbe
        ax.xaxis.set_major_locator(FixedLocator(n))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: (f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms").replace(".", ",")))
        for axe in (ax.xaxis, ax.yaxis):
            axe.set_minor_locator(NullLocator())
        ax.grid(True, axis="y", color=trait, linewidth=0.8)
        ax.tick_params(length=0, pad=6)
        for cote in ("top", "right", "left"):
            ax.spines[cote].set_visible(False)
        ax.spines["bottom"].set_color(trait)
        ax.legend(loc="upper left", labelcolor=encre, handlelength=1.8,
                  facecolor=fond, edgecolor=fond, framealpha=1)  # fond plein : la grille ne barre pas le texte
    fig.tight_layout(w_pad=4)
    fig.savefig(chemin, dpi=180, facecolor=fond)
    plt.close(fig)
    print(f"Graphique : {chemin.name}")


def main():
    """Exécuter les tailles prévues et enregistrer bench_results.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="s'arrêter à 1 000 restaurants")
    args = parser.parse_args()
    racine = Path(__file__).resolve().parent
    positions = np.random.default_rng(43).uniform((0, 0), (1000, 700), (1000, 2))
    lignes = []
    print("N | Carte (ms) | Voronoï | Delaunay | Localisation | 1000 recherches | Force brute", flush=True)
    with (racine / "bench_results.csv").open("w", encoding="utf-8", newline="") as fichier:
        sortie = csv.writer(fichier)
        sortie.writerow(COLUMNS)
        for n in SIZES[:3] if args.quick else SIZES:
            restaurants = (charger_csv(racine / "data" / "restaurants.csv") if n == 30
                           else generer_restaurants(n, 42))
            # Amorcer VTK évite d'attribuer son initialisation à la première taille.
            Carte(restaurants[:1])
            phases, durees = [], []
            carte = None
            for _ in range(3):
                carte = None
                debut = perf_counter()
                carte = Carte(restaurants)
                durees.append((perf_counter() - debut) * 1000)
                phases.append(carte.temps_ms)
            recherche_ms, indices = mesurer(lambda: np.array([carte.plus_proche(float(x), float(y))
                                                              for x, y in positions]))
            brute_ms, attendus = mesurer(lambda: brute_numpy(carte.sites, positions))
            obtenues = np.sum((carte.sites[indices] - positions) ** 2, axis=1)
            minimales = np.sum((carte.sites[attendus] - positions) ** 2, axis=1)
            assert np.all(obtenues <= minimales * (1 + 1e-9)), "Recherche différente de la force brute"
            assert carte.voronoi.GetNumberOfPolys() == n, "Une cellule par restaurant attendue"
            assert carte.triangles.shape[1] == 3 and len(carte.triangles) > 0
            ligne = [n, median(durees), *(median(phase[k] for phase in phases)
                                         for k in ("voronoi", "delaunay", "localisation")), recherche_ms, brute_ms]
            lignes.append(ligne)
            sortie.writerow([n] + [f"{v:.4f}" for v in ligne[1:]])
            fichier.flush()
            print(f"{n:6d} | " + " | ".join(f"{v:10.3f}" for v in ligne[1:]), flush=True)
    tracer(lignes, racine / "bench_temps.png")


if __name__ == "__main__":
    main()
