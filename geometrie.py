"""Moteur réutilisable de recherche de restaurants et de géométrie 2D avec VTK.

Les indices publics désignent toujours la liste locale ``Carte.restaurants``.
Les doublons exacts partagent une tuile et forment une clique de voisins ; les
triangles utilisent le premier restaurant de chaque site. Ne pas modifier les
listes ou objets VTK d'une carte après sa construction.
"""

from __future__ import annotations

import csv
import heapq
import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


@dataclass(frozen=True)
class Restaurant:
    """Restaurant, coordonnées cartésiennes et attributs de filtrage."""

    id: int
    nom: str
    x: float
    y: float
    cuisine: str = "Autre"
    prix: int = 0
    note: float | None = None
    avis: int | None = None

    @property
    def prix_txt(self) -> str:
        """Prix sous forme de symboles euro, ou « ? » si inconnu."""
        return "€" * self.prix if self.prix else "?"


class ErreurDonnees(ValueError):
    """Données absentes, mal formées ou incompatibles avec la carte."""


def charger_csv(chemin) -> list[Restaurant]:
    """Lire un CSV UTF-8 (BOM accepté), séparé par des virgules ou des « ; ».

    ID, nom, X et Y sont obligatoires, sans distinction de casse ou d'espaces.
    Les champs facultatifs vides prennent les valeurs par défaut de Restaurant.
    """
    restaurants = []
    try:
        with open(chemin, encoding="utf-8-sig", newline="") as fichier:
            extrait = fichier.read(8192)
            fichier.seek(0)
            entete = extrait.splitlines()[0] if extrait else ""
            separateur = ";" if entete.count(";") > entete.count(",") else ","
            lecteur = csv.DictReader(fichier, delimiter=separateur, strict=True)
            if lecteur.fieldnames is None:
                raise ErreurDonnees("Le fichier CSV est vide : aucun restaurant.")
            colonnes = ["".join(c.casefold().split()) for c in lecteur.fieldnames]
            if len(set(colonnes)) != len(colonnes):
                raise ErreurDonnees("Ligne 1 : noms de colonnes répétés.")
            manquantes = {"id", "nom", "x", "y"} - set(colonnes)
            if manquantes:
                raise ErreurDonnees("Ligne 1 : colonnes obligatoires manquantes : "
                                    + ", ".join(sorted(manquantes)) + ".")
            lecteur.fieldnames = colonnes
            identifiants = set()
            for ligne in lecteur:
                numero = lecteur.line_num
                try:
                    if None in ligne or any(ligne[c] is None for c in colonnes):
                        raise ValueError("nombre de champs incorrect")
                    ligne = {c: v.strip() for c, v in ligne.items()}
                    identifiant = int(ligne["id"])
                    if identifiant in identifiants:
                        raise ValueError("ID déjà utilisé")
                    nom = ligne["nom"]
                    if not nom:
                        raise ValueError("nom vide")
                    x, y = float(ligne["x"]), float(ligne["y"])
                    if not (math.isfinite(x) and math.isfinite(y)):
                        raise ValueError("X et Y doivent être finis")
                    texte_prix = ligne.get("prix", "")
                    if not texte_prix:
                        prix = 0
                    elif set(texte_prix) == {"€"}:
                        prix = len(texte_prix)
                    else:
                        prix = int(texte_prix)
                    if not 0 <= prix <= 4:
                        raise ValueError("prix attendu entre 0 et 4, ou de € à €€€€")
                    texte_note = ligne.get("note", "")
                    note = float(texte_note.replace(",", ".")) if texte_note else None
                    if note is not None and not (math.isfinite(note) and 0 <= note <= 5):
                        raise ValueError("note attendue entre 0 et 5")
                    texte_avis = ligne.get("avis", "") or ligne.get("nombre_avis", "")
                    avis = int(texte_avis) if texte_avis else None
                    if avis is not None and avis < 0:
                        raise ValueError("nombre d'avis négatif")
                    restaurants.append(Restaurant(identifiant, nom, x, y,
                                                  ligne.get("type") or "Autre",
                                                  prix, note, avis))
                    identifiants.add(identifiant)
                except (ValueError, OverflowError) as erreur:
                    raise ErreurDonnees(f"Ligne {numero} : valeur invalide ({erreur}).") from erreur
    except (OSError, UnicodeError, csv.Error) as erreur:
        numero = lecteur.line_num if "lecteur" in locals() else 1
        raise ErreurDonnees(f"Lecture du CSV, ligne {numero} : {erreur}.") from erreur
    if not restaurants:
        raise ErreurDonnees("Le fichier ne contient aucun restaurant.")
    return restaurants


def _bornes_valides(bornes):
    valeurs = np.asarray(bornes, dtype=np.float64)
    if (valeurs.shape != (4,) or not np.isfinite(valeurs).all()
            or valeurs[0] >= valeurs[1] or valeurs[2] >= valeurs[3]):
        raise ErreurDonnees("Le cadre doit contenir quatre bornes finies : xmin < xmax, ymin < ymax.")
    return tuple(float(v) for v in valeurs)


# Cuisines du fichier fourni, avec leurs fréquences, et des mots pour composer des noms plausibles.
_CUISINES = {
    "Français": (8, ("Le Bistrot", "Le Comptoir", "Le Petit Zinc", "La Cocotte", "La Table", "Le Gourmet")),
    "Italien": (5, ("La Trattoria", "L'Osteria", "La Pizzeria", "Pasta Mia", "Napoli", "La Piazza")),
    "Végétarien": (3, ("Le Potager", "Le Jardin", "La Graine", "Green Bowl", "La Pousse")),
    "Japonais": (2, ("Sushi Bar", "Ramen Ya", "Sakura", "L'Izakaya", "Tokyo Kitchen")),
    "Asiatique": (2, ("Le Wok", "Le Lotus", "Le Bambou", "Le Mandarin", "Saigon")),
    "Indien": (2, ("Le Curry", "Masala", "Bombay", "Le Tandoori", "Taj")),
    "Oriental": (2, ("Le Falafel", "Le Cèdre", "L'Atlas", "La Médina", "Saveurs d'Orient")),
    "Fruits de mer": (2, ("La Marée", "L'Écailler", "Le Homard", "Océan")),
    "Mexicain": (2, ("La Cantina", "Taco Loco", "Azteca", "Casa Latina")),
    "Burgers": (1, ("Burger Club", "Smash House", "Le Diner")),
    "Grillades": (1, ("Le Grill", "La Braise", "Le Brasero")),
}
_PRENOMS = ("Marco", "Aïcha", "Lucien", "Mei", "Rosa", "Karim", "Paulette", "Inès", "Hugo", "Lina", "Yann", "Sofia")
_LIEUX = ("", "", " du Port", " des Halles", " Saint-Jean", " de la Gare", " d'Or", " du Marché", " du Centre")


def generer_restaurants(n: int, graine: int = 0,
                       bornes=(0.0, 1000.0, 0.0, 700.0)) -> list[Restaurant]:
    """Générer n restaurants reproductibles, uniformes dans le rectangle donné (extension D).

    Cuisines tirées avec les fréquences du fichier fourni, noms composés, prix de 1 à 4 €,
    note autour de 4,2 (écart type 0,35, bornée à [2,5 ; 5]) et nombre d'avis.
    """
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ErreurDonnees("Le nombre de restaurants doit être un entier positif ou nul.")
    xmin, xmax, ymin, ymax = _bornes_valides(bornes)
    rng = np.random.default_rng(graine)
    positions = rng.uniform((xmin, ymin), (xmax, ymax), (n, 2))
    noms_cuisines = list(_CUISINES)
    poids = np.array([_CUISINES[c][0] for c in noms_cuisines], dtype=float)
    cuisines = rng.choice(len(noms_cuisines), size=n, p=poids / poids.sum())
    prix = rng.choice(4, size=n, p=(0.3, 0.45, 0.2, 0.05)) + 1
    notes = np.clip(np.round(rng.normal(4.2, 0.35, n), 1), 2.5, 5.0)
    avis = np.round(rng.lognormal(4.6, 0.8, n)).astype(int)
    restaurants = []
    for i, (x, y) in enumerate(positions):
        cuisine = noms_cuisines[cuisines[i]]
        if rng.random() < 0.25:
            nom = f"Chez {_PRENOMS[rng.integers(len(_PRENOMS))]}"
        else:
            mots = _CUISINES[cuisine][1]
            nom = mots[rng.integers(len(mots))] + _LIEUX[rng.integers(len(_LIEUX))]
        restaurants.append(Restaurant(i, nom, float(x), float(y), cuisine, int(prix[i]),
                                      float(notes[i]), int(avis[i])))
    return restaurants


def cadre(restaurants, marge: float = 0.08) -> tuple[float, float, float, float]:
    """Rectangle des sites avec une marge ; cadre par défaut si la liste est vide."""
    if not math.isfinite(marge) or marge < 0:
        raise ErreurDonnees("La marge doit être finie et positive ou nulle.")
    positions = np.array([(r.x, r.y) for r in restaurants], dtype=np.float64)
    if positions.size == 0:
        return (0.0, 1000.0, 0.0, 700.0)
    if not np.isfinite(positions).all():
        raise ErreurDonnees("Les coordonnées des restaurants doivent être finies.")
    bas, haut = positions.min(axis=0), positions.max(axis=0)
    etendue = haut - bas
    # Une droite ou un site isolé doit aussi donner un rectangle non dégénéré.
    largeur = max(float(etendue.max()), 1.0)
    ajout = np.where(etendue > 0, etendue * marge, largeur * max(marge, 0.08))
    return (float(bas[0] - ajout[0]), float(haut[0] + ajout[0]),
            float(bas[1] - ajout[1]), float(haut[1] + ajout[1]))


def _polydata(positions):
    points = vtk.vtkPoints()
    xyz = np.zeros((len(positions), 3), dtype=np.float64)
    xyz[:, :2] = positions
    points.SetData(numpy_to_vtk(xyz, deep=True))
    resultat = vtk.vtkPolyData()
    resultat.SetPoints(points)
    return resultat


def _cellules(longueurs):
    offsets = np.r_[0, np.cumsum(longueurs, dtype=np.int64)]
    cellules = vtk.vtkCellArray()
    cellules.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                     numpy_to_vtkIdTypeArray(np.arange(offsets[-1], dtype=np.int64), deep=True))
    return cellules


def _decouper(polygone, bornes):
    """Découpe rectangulaire seulement : les cellules sont construites par VTK."""
    sommets = [tuple(p) for p in polygone]
    for axe, limite, signe in ((0, bornes[0], 1), (0, bornes[1], -1),
                               (1, bornes[2], 1), (1, bornes[3], -1)):
        sortie = []
        if not sommets:
            break
        precedent = sommets[-1]
        dedans_avant = signe * (precedent[axe] - limite) >= 0
        for courant in sommets:
            dedans = signe * (courant[axe] - limite) >= 0
            if dedans != dedans_avant:
                t = (limite - precedent[axe]) / (courant[axe] - precedent[axe])
                p = [precedent[k] + t * (courant[k] - precedent[k]) for k in (0, 1)]
                p[axe] = limite
                sortie.append(tuple(p))
            if dedans:
                sortie.append(courant)
            precedent, dedans_avant = courant, dedans
        sommets = sortie
    return np.asarray(sommets, dtype=np.float64).reshape(-1, 2)


class Carte:
    """Modèle géométrique construit une fois, sans opération de modification.

    Les doublons exacts gardent leurs indices, partagent la même cellule et sont
    voisins entre eux et de tous les restaurants des sites voisins. La sortie
    Voronoï répète alors cette cellule pour chaque restaurant (aires superposées).
    Les bornes explicites doivent contenir tous les sites.
    """

    def __init__(self, restaurants: Sequence[Restaurant], bornes: tuple | None = None):
        self.restaurants = list(restaurants)
        self.n = len(self.restaurants)
        self.bornes = _bornes_valides(cadre(self.restaurants) if bornes is None else bornes)
        self.sites = np.array([(r.x, r.y) for r in self.restaurants],
                              dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(self.sites).all():
            raise ErreurDonnees("Les coordonnées des restaurants doivent être finies.")
        xmin, xmax, ymin, ymax = self.bornes
        if np.any((self.sites < (xmin, ymin)) | (self.sites > (xmax, ymax))):
            raise ErreurDonnees("Le cadre doit contenir tous les restaurants.")
        self.sites.flags.writeable = False
        self._origine = np.array((xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2))
        self._echelle = max(xmax - xmin, ymax - ymin)
        self._normalises = (self.sites - self._origine) / self._echelle
        self._cadre_normalise = ((xmin - self._origine[0]) / self._echelle,
                                 (xmax - self._origine[0]) / self._echelle,
                                 (ymin - self._origine[1]) / self._echelle,
                                 (ymax - self._origine[1]) / self._echelle)
        _, premiers, inverse = np.unique(self.sites, axis=0, return_index=True, return_inverse=True)
        ordre = np.argsort(premiers)
        self._representants = premiers[ordre]
        self._groupe = np.argsort(ordre)[inverse]
        self._groupes = [[] for _ in premiers]
        for i, groupe in enumerate(self._groupe):
            self._groupes[groupe].append(i)
        self._uniques = self._normalises[self._representants]
        self._adjacence = [set() for _ in self.restaurants]
        self._frontieres = {}
        self._cercles = None
        self.temps_ms = {}

        debut = perf_counter()
        self.points = _polydata(self.sites)
        self.points.SetVerts(_cellules(np.ones(self.n, dtype=np.int64)))
        noms = vtk.vtkStringArray()
        noms.SetName("nom")
        for r in self.restaurants:
            noms.InsertNextValue(r.nom)
        self.points.GetPointData().AddArray(noms)
        indices = numpy_to_vtk(np.arange(self.n, dtype=np.int32), deep=True)
        indices.SetName("indice")
        self.points.GetPointData().AddArray(indices)
        self._localisateur = vtk.vtkStaticPointLocator2D()
        if self.n:
            self._localisateur.SetDataSet(self.points)
            self._localisateur.BuildLocator()
        self.temps_ms["localisation"] = (perf_counter() - debut) * 1000

        debut = perf_counter()
        self._construire_voronoi()
        self.temps_ms["voronoi"] = (perf_counter() - debut) * 1000
        debut = perf_counter()
        self._construire_delaunay()
        self.temps_ms["delaunay"] = (perf_counter() - debut) * 1000

    def _construire_voronoi(self):
        if self.n:
            # Les fantômes sont plus loin de tout point du cadre que tout site réel.
            # Leur boîte contient le cadre, même après un filtrage dans un seul coin.
            fantomes = np.array(((-4., -4.), (4., -4.), (4., 4.), (-4., 4.)))
            entree = _polydata(np.vstack((self._uniques, fantomes)))
            vor = vtk.vtkVoronoiFlower2D()
            vor.SetInputData(entree)
            vor.SetOutputTypeToVoronoi()
            vor.SetGenerateCellScalarsToPointIds()
            vor.SetPadding(0.01)
            vor.SetMergePoints(False)
            vor.SetPruneTolerance(0.0)
            vor.SetValidate(True)
            vor.Update()
            sortie = vor.GetOutput()
            offsets = vtk_to_numpy(sortie.GetPolys().GetOffsetsArray())
            connexion = vtk_to_numpy(sortie.GetPolys().GetConnectivityArray())
            sommets = vtk_to_numpy(sortie.GetPoints().GetData())[connexion, :2]
            ids = vtk_to_numpy(sortie.GetCellData().GetScalars())
            b = self._cadre_normalise
            hors = np.any((sommets < (b[0], b[2])) | (sommets > (b[1], b[3])), axis=1)
            a_decouper = np.logical_or.reduceat(hors, offsets[:-1])
            suivants = np.roll(sommets, -1, axis=0)
            suivants[offsets[1:] - 1] = sommets[offsets[:-1]]
            aires = np.add.reduceat(sommets[:, 0] * suivants[:, 1]
                                   - sommets[:, 1] * suivants[:, 0], offsets[:-1])
            tuiles = [None] * len(self._uniques)
            for cellule, indice in enumerate(ids):
                if indice >= len(tuiles):
                    continue
                tuile = sommets[offsets[cellule]:offsets[cellule + 1]]
                if a_decouper[cellule]:
                    tuile = _decouper(tuile, b)
                if len(tuile) < 3:
                    raise ErreurDonnees("VTK a produit une cellule de Voronoï dégénérée.")
                tuiles[indice] = tuile if aires[cellule] >= 0 else tuile[::-1]
            if any(t is None for t in tuiles):
                raise ErreurDonnees("VTK n'a pas produit une cellule pour chaque site.")
            self._tuiles = tuiles
            longueurs = [len(tuiles[g]) for g in self._groupe]
            positions = np.concatenate([tuiles[g] for g in self._groupe])
            positions = positions * self._echelle + self._origine
            # Fixer les coordonnées du cadre évite tout dépassement d'un ulp.
            positions[:, 0] = np.clip(positions[:, 0], self.bornes[0], self.bornes[1])
            positions[:, 1] = np.clip(positions[:, 1], self.bornes[2], self.bornes[3])
        else:
            self._tuiles, longueurs, positions = [], [], np.empty((0, 2))
        self.voronoi = _polydata(positions)
        self.voronoi.SetPolys(_cellules(longueurs))
        indices = numpy_to_vtk(np.arange(self.n, dtype=np.int32), deep=True)
        indices.SetName("indice")
        self.voronoi.GetCellData().SetScalars(indices)

    def _construire_delaunay(self):
        nombre = len(self._uniques)
        triangles = np.empty((0, 3), dtype=np.int64)
        aretes = np.empty((0, 2), dtype=np.int64)
        if nombre >= 2:
            etendue = np.ptp(self._uniques, axis=0)
            axe = int(np.argmax(etendue))
            ordre = np.argsort(self._uniques[:, axe])
            vecteur = self._uniques[ordre[-1]] - self._uniques[ordre[0]]
            delta = self._uniques - self._uniques[ordre[0]]
            produits = delta[:, 0] * vecteur[1] - delta[:, 1] * vecteur[0]
            colineaires = np.max(np.abs(produits)) <= 8 * np.finfo(float).eps * np.dot(vecteur, vecteur)
            if colineaires:
                aretes = np.column_stack((ordre[:-1], ordre[1:]))
            else:
                # Flower borne son dual : Delaunay2D indépendant garde aussi les
                # triangles dont le centre circonscrit est hors du cadre affiché.
                dela = vtk.vtkDelaunay2D()
                dela.SetInputData(_polydata(self._uniques))
                dela.SetTolerance(0.0)
                dela.SetOffset(1000.0)
                dela.Update()
                sortie = dela.GetOutput()
                connexion = vtk_to_numpy(sortie.GetPolys().GetConnectivityArray())
                triangles = connexion.reshape(-1, 3).copy()
                aretes = np.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, ::2]))
                aretes = np.unique(np.sort(aretes, axis=1), axis=0)
        self.triangles = self._representants[triangles]
        self.delaunay = _polydata(self.sites)
        cellules = vtk.vtkCellArray()
        cellules.SetData(numpy_to_vtkIdTypeArray(np.arange(len(triangles) + 1, dtype=np.int64) * 3, deep=True),
                         numpy_to_vtkIdTypeArray(self.triangles.ravel(), deep=True))
        self.delaunay.SetPolys(cellules)
        for groupe in self._groupes:
            for i in groupe:
                self._adjacence[i].update(j for j in groupe if j != i)
        for a, b in aretes:
            for i in self._groupes[a]:
                self._adjacence[i].update(self._groupes[b])
            for j in self._groupes[b]:
                self._adjacence[j].update(self._groupes[a])
        self.aretes = np.array([(i, j) for i, voisins in enumerate(self._adjacence)
                                for j in sorted(voisins) if i < j], dtype=np.int64).reshape(-1, 2)
        self.triangles.flags.writeable = False
        self.aretes.flags.writeable = False

    def _verifier_indice(self, i):
        if not isinstance(i, (int, np.integer)) or not 0 <= i < self.n:
            raise IndexError(f"Indice de restaurant hors de la carte : {i}.")

    def plus_proche(self, x: float, y: float) -> int | None:
        """Indice du site le plus proche, même hors cadre ; None si la carte est vide.

        Une égalité de distances peut renvoyer n'importe lequel des ex æquo.
        La recherche se fait sur les points en double précision, sans tolérance
        de cellule susceptible de déplacer les frontières.
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ErreurDonnees("La position de recherche doit être finie.")
        return self._localisateur.FindClosestPoint(x, y, 0.0) if self.n else None

    def voisins(self, i: int) -> list[int]:
        """Indices triés des voisins de Delaunay et des restaurants au même site."""
        self._verifier_indice(i)
        return sorted(self._adjacence[i])

    def voisins_ordre(self, i: int, ordre: int = 2) -> dict[int, int]:
        """Distances dans le graphe, jusqu'à l'anneau ordre, sans le site de départ."""
        self._verifier_indice(i)
        if not isinstance(ordre, (int, np.integer)) or ordre < 0:
            raise ValueError("L'ordre doit être un entier positif ou nul.")
        distances = {i: 0}
        attente = deque([i])
        while attente:
            courant = attente.popleft()
            if distances[courant] >= ordre:
                continue
            for j in self.voisins(courant):
                if j not in distances:
                    distances[j] = distances[courant] + 1
                    attente.append(j)
        del distances[i]
        return distances

    def tuile(self, i: int) -> np.ndarray:
        """Copie des sommets de la cellule de i, en sens trigonométrique."""
        self._verifier_indice(i)
        return self._tuiles[self._groupe[i]] * self._echelle + self._origine

    def frontiere(self, i: int, j: int) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Segment commun, ou None pour un contact ponctuel ou absent du cadre.

        Deux sites confondus partagent toute leur tuile, sans frontière séparatrice.
        Une arête de Delaunay du bord peut avoir son dual entièrement hors cadre.
        """
        self._verifier_indice(i)
        self._verifier_indice(j)
        cle = (min(i, j), max(i, j))
        if cle in self._frontieres:
            return self._frontieres[cle]
        resultat = None
        if j in self._adjacence[i] and self._groupe[i] != self._groupe[j]:
            a, b = self._normalises[[i, j]]
            normale = b - a
            normale /= np.linalg.norm(normale)
            milieu = a + (b - a) / 2
            tuile = self._tuiles[self._groupe[i]]
            sur_ligne = np.abs((tuile - milieu) @ normale) <= 128 * np.finfo(float).eps
            if np.count_nonzero(sur_ligne) >= 2:
                direction = np.array((-normale[1], normale[0]))
                t = (tuile[sur_ligne] - milieu) @ direction
                if np.ptp(t) > 128 * np.finfo(float).eps:
                    p, q = (milieu + np.array((t.min(), t.max()))[:, None] * direction) * self._echelle + self._origine
                    resultat = (tuple(float(v) for v in p), tuple(float(v) for v in q))
        self._frontieres[cle] = resultat
        return resultat

    def segments_voronoi(self) -> list[tuple[tuple[float, float], tuple[float, float], int, int]]:
        """Frontières internes non nulles, une fois par paire de sites distincts."""
        resultat = []
        for i, j in self.aretes:
            if i != self._representants[self._groupe[i]] or j != self._representants[self._groupe[j]]:
                continue
            frontiere = self.frontiere(int(i), int(j))
            if frontiere is not None:
                resultat.append((*frontiere, int(i), int(j)))
        return resultat

    def cercles_circonscrits(self) -> np.ndarray:
        """Tableau (cx, cy, r) des triangles, dans le même ordre que triangles.

        Les centres hors du rectangle ne sont pas des sommets des cellules découpées.
        """
        if self._cercles is None:
            a, b, c = self._normalises[self.triangles].transpose(1, 0, 2)
            u, v = b - a, c - a
            determinant = 2 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
            u2, v2 = np.sum(u * u, axis=1), np.sum(v * v, axis=1)
            delta = np.column_stack(((u2 * v[:, 1] - v2 * u[:, 1]) / determinant,
                                     (u[:, 0] * v2 - v[:, 0] * u2) / determinant))
            centres = (a + delta) * self._echelle + self._origine
            self._cercles = np.column_stack((centres, np.linalg.norm(delta, axis=1) * self._echelle))
            self._cercles.flags.writeable = False
        return self._cercles.copy()

    def cercles(self, i: int) -> list[tuple[float, float, float]]:
        """Cercles vides des triangles incidents au site de i (sa fleur)."""
        self._verifier_indice(i)
        representant = self._representants[self._groupe[i]]
        incidents = np.any(self.triangles == representant, axis=1)
        return [tuple(float(v) for v in cercle) for cercle in self.cercles_circonscrits()[incidents]]

    def distance(self, i: int, x: float, y: float) -> float:
        """Distance euclidienne du restaurant i à la position donnée."""
        self._verifier_indice(i)
        return math.hypot(self.sites[i, 0] - x, self.sites[i, 1] - y)

    def traversees(self, x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float, int, int]]:
        """Franchissements exacts, strictement entre les extrémités du déplacement.

        L'enveloppe inférieure des distances affines évite tout échantillonnage :
        le terme quadratique du déplacement est identique pour tous les sites.
        Les cellules touchées en un seul point ne comptent pas comme traversées.
        La recherche reste valable hors du rectangle, comme plus_proche.
        """
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            raise ErreurDonnees("Les extrémités du déplacement doivent être finies.")
        if self.n < 2 or (x0 == x1 and y0 == y1):
            return []
        depart = (np.array((x0, y0)) - self._origine) / self._echelle
        direction = np.array((x1 - x0, y1 - y0)) / self._echelle
        delta = self._uniques - depart
        intercepts = np.einsum("ij,ij->i", delta, delta)
        pentes = -2 * (delta @ direction)
        ordre = np.lexsort((self._representants, intercepts, -pentes))
        enveloppe, debuts = [], []
        for j in ordre:
            if enveloppe and pentes[j] == pentes[enveloppe[-1]]:
                continue
            t = -math.inf
            while enveloppe:
                precedent = enveloppe[-1]
                t = (intercepts[j] - intercepts[precedent]) / (pentes[precedent] - pentes[j])
                if t > debuts[-1]:
                    break
                enveloppe.pop()
                debuts.pop()
            enveloppe.append(j)
            debuts.append(t if len(enveloppe) > 1 else -math.inf)
        return [(float(x0 + t * (x1 - x0)), float(y0 + t * (y1 - y0)),
                 int(self._representants[enveloppe[k - 1]]), int(self._representants[enveloppe[k]]))
                for k, t in enumerate(debuts) if 0 < t < 1]

    def coloration(self, k: int = 6) -> np.ndarray:
        """Coloration DSATUR déterministe ; dépasse k si le glouton le nécessite.

        k est un maximum souhaité, jamais au prix de voisins de même couleur
        (par exemple une clique de plus de k restaurants au même site).
        """
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("Le nombre de couleurs souhaité doit être au moins 1.")
        couleurs = np.full(self.n, -1, dtype=np.int64)
        interdites = [set() for _ in self.restaurants]
        tas = [(0, -len(v), i) for i, v in enumerate(self._adjacence)]
        heapq.heapify(tas)
        while tas:
            saturation, _, i = heapq.heappop(tas)
            if couleurs[i] >= 0 or -saturation != len(interdites[i]):
                continue
            couleur = 0
            while couleur in interdites[i]:
                couleur += 1
            couleurs[i] = couleur
            for j in self._adjacence[i]:
                if couleurs[j] < 0 and couleur not in interdites[j]:
                    interdites[j].add(couleur)
                    heapq.heappush(tas, (-len(interdites[j]), -len(self._adjacence[j]), j))
        return couleurs
