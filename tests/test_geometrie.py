"""Tests autonomes : python tests/test_geometrie.py ; également compatibles pytest."""

import math
import sys
import tempfile
import traceback
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from geometrie import Carte, ErreurDonnees, Restaurant, cadre, charger_csv, generer_restaurants


@lru_cache
def reelle():
    return Carte(charger_csv(RACINE / "data" / "restaurants.csv"))


@lru_cache
def aleatoire():
    return Carte(generer_restaurants(2000, 17))


def points(positions):
    return [Restaurant(100 + i, f"R{i}", float(x), float(y)) for i, (x, y) in enumerate(positions)]


def aire(tuile):
    p = tuile - tuile[0]
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)) / 2)


def verifier_tuiles(c):
    xmin, xmax, ymin, ymax = c.bornes
    total, uniques = 0., set()
    for i, site in enumerate(c.sites):
        t = c.tuile(i)
        assert t.shape[1] == 2 and len(t) >= 3 and np.isfinite(t).all()
        assert np.all(t >= np.array((xmin, ymin)) - 1e-9)
        assert np.all(t <= np.array((xmax, ymax)) + 1e-9)
        assert aire(t) > 0
        cote, delta = np.roll(t, -1, axis=0) - t, site - t
        croix = cote[:, 0] * delta[:, 1] - cote[:, 1] * delta[:, 0]
        assert np.all(croix >= -1e-8 * max(xmax - xmin, ymax - ymin)), (i, croix)
        proches = [c.plus_proche(float(x), float(y)) for x, y in t]
        distances = np.sum((t - site) ** 2, axis=1)
        minimales = np.sum((t - c.sites[proches]) ** 2, axis=1)
        assert np.all(distances <= minimales * (1 + 1e-8) + 1e-8), i
        # Les restaurants confondus partagent leur tuile : l'union se compte une fois.
        if tuple(site) not in uniques:
            total += aire(t)
            uniques.add(tuple(site))
    assert math.isclose(total, (xmax - xmin) * (ymax - ymin), rel_tol=1e-6), total
    assert c.voronoi.GetNumberOfPolys() == c.n and c.points.GetNumberOfVerts() == c.n
    assert c.points.GetPointData().GetAbstractArray("nom").GetNumberOfValues() == c.n
    assert np.array_equal(vtk_to_numpy(c.points.GetPointData().GetArray("indice")), np.arange(c.n))
    assert np.array_equal(vtk_to_numpy(c.voronoi.GetCellData().GetArray("indice")), np.arange(c.n))


def verifier_recherches(c, positions):
    debut = perf_counter()
    indices = [c.plus_proche(float(x), float(y)) for x, y in positions]
    microsecondes = (perf_counter() - debut) * 1e6 / len(positions)
    for p, i in zip(positions, indices):
        delta = c.sites - p
        distances = np.einsum("ij,ij->i", delta, delta)
        assert distances[i] <= distances.min() * (1 + 1e-9), (p, i, distances[i], distances.min())
    return microsecondes


def test_csv_reel_et_curry_palace():
    c = reelle()
    assert c.n == 30 and c.restaurants[c.plus_proche(430, 270)].nom == "Curry Palace"
    assert c.restaurants[0].prix_txt == "€€"
    assert Restaurant(1, "Sans prix", 0, 0).prix_txt == "?"


def test_recherche_20000_positions_reelles():
    c = reelle()
    b = c.bornes
    positions = np.random.default_rng(1).uniform((b[0] - 500, b[2] - 500), (b[1] + 500, b[3] + 500), (20000, 2))
    assert verifier_recherches(c, positions) < 50


def test_recherche_20000_positions_2000_restaurants():
    positions = np.random.default_rng(3).uniform((-100, -100), (1100, 800), (20000, 2))
    assert verifier_recherches(aleatoire(), positions) < 50


def test_recherche_pres_des_frontieres():
    c, positions = reelle(), []
    for p, q, i, j in c.segments_voronoi():
        milieu = (np.array(p) + q) / 2
        direction = c.sites[j] - c.sites[i]
        direction /= np.linalg.norm(direction)
        for epsilon in (1e-2, 1e-5, 1e-8):
            positions.extend((milieu + epsilon * direction, milieu - epsilon * direction))
    verifier_recherches(c, positions)


def test_cercles_vides():
    for c in (reelle(), aleatoire()):
        cercles = c.cercles_circonscrits()
        assert cercles.shape == (len(c.triangles), 3)
        for triangle, (cx, cy, rayon) in zip(c.triangles, cercles):
            distances = np.sum((c.sites - (cx, cy)) ** 2, axis=1)
            assert np.all(distances >= rayon ** 2 - max(1e-7, rayon ** 2 * 1e-9))
            assert np.allclose(distances[triangle], rayon ** 2, rtol=1e-9, atol=1e-7)
        for i in range(min(c.n, 30)):
            assert len(c.cercles(i)) == np.count_nonzero(np.any(c.triangles == i, axis=1))


def test_dualite_position_generale():
    # Le cadre contient les centres circonscrits, donc toutes les frontières duales.
    c = Carte(points(np.random.default_rng(123).uniform(-1, 1, (15, 2))), (-100., 100., -100., 100.))
    for i in range(c.n):
        for j in range(i + 1, c.n):
            segment = c.frontiere(i, j)
            assert (j in c.voisins(i)) == (segment is not None), (i, j)
            if segment is not None:
                p, q = np.array(segment)
                assert np.linalg.norm(q - p) > 0 and c.frontiere(j, i) == segment
                assert np.allclose(np.linalg.norm(np.array((p, q)) - c.sites[i], axis=1),
                                   np.linalg.norm(np.array((p, q)) - c.sites[j], axis=1))


def test_aires_et_sites_reels_aleatoires():
    verifier_tuiles(reelle())
    verifier_tuiles(aleatoire())


def test_filtre_et_coin_du_cadre():
    c = reelle()
    verifier_tuiles(Carte([r for r in c.restaurants if r.cuisine == "Italien"], c.bornes))
    coin = Carte(points([(1, 1), (1.001, 1), (2, 2), (1, 3)]), (0., 1000., 0., 700.))
    verifier_tuiles(coin)
    verifier_recherches(coin, np.random.default_rng(2).uniform((0, 0), (1000, 700), (1000, 2)))


def test_carte_vide():
    c = Carte([])
    assert c.n == 0 and c.plus_proche(0, 0) is None and c.sites.shape == (0, 2)
    assert c.triangles.shape == (0, 3) and c.aretes.shape == (0, 2)
    assert c.voronoi.GetNumberOfCells() == c.delaunay.GetNumberOfCells() == 0
    assert c.segments_voronoi() == [] and c.traversees(0, 0, 1, 1) == []
    assert c.cercles_circonscrits().shape == (0, 3) and c.coloration().shape == (0,)


def test_un_restaurant():
    c = Carte(points([(1, 2)]), (-10., 10., -20., 20.))
    verifier_tuiles(c)
    assert len(c.tuile(0)) == 4 and c.voisins(0) == []
    assert c.plus_proche(-1000, 5000) == 0 and c.cercles(0) == []
    assert c.voisins_ordre(0) == {} and c.distance(0, 4, 6) == 5


def test_deux_restaurants():
    c = Carte(points([(0, 0), (2, 0)]), (-1., 3., -1., 1.))
    verifier_tuiles(c)
    assert c.voisins(0) == [1] and c.voisins(1) == [0]
    assert np.array_equal(c.aretes, [[0, 1]]) and c.triangles.shape == (0, 3)
    frontiere = np.array(c.frontiere(0, 1))
    assert np.allclose(frontiere[:, 0], 1) and np.allclose(sorted(frontiere[:, 1]), [-1, 1])
    assert c.traversees(0, 0, 2, 0) == [(1., 0., 0, 1)]


def test_points_colineaires():
    for positions in ([(-3, -5), (-1, -1), (0, 1), (2, 5)],
                      [(0, 3), (0, 0), (0, 1), (0, 2)], [(3, 0), (0, 0), (1, 0), (2, 0)]):
        c = Carte(points(positions))
        verifier_tuiles(c)
        ordre = np.argsort(np.array(positions)[:, int(np.argmax(np.ptp(positions, axis=0)))])
        attendues = {tuple(sorted((int(a), int(b)))) for a, b in zip(ordre[:-1], ordre[1:])}
        assert set(map(tuple, c.aretes)) == attendues and len(c.triangles) == 0
        assert all(c.voisins(i) for i in range(c.n))


def test_quasi_doublons():
    for decalage in (0., 1e6):
        positions = np.array([(0, 0), (.001, 0), (1, 0), (0, 1), (1, 1)]) + decalage
        c = Carte(points(positions))
        verifier_tuiles(c)
        assert 1 in c.voisins(0) and all(c.voisins(i) for i in range(c.n))
        assert all(c.plus_proche(x, y) == i for i, (x, y) in enumerate(positions))
        verifier_recherches(c, np.random.default_rng(6).uniform(-.1, 1.1, (1000, 2)) + decalage)


def test_doublons_groupes():
    c = Carte(points([(0, 0), (0, 0), (1, 0), (0, 1), (1, 1)]))
    verifier_tuiles(c)
    assert np.array_equal(c.tuile(0), c.tuile(1)) and c.plus_proche(0, 0) in (0, 1)
    assert 1 in c.voisins(0) and 0 in c.voisins(1)
    assert set(c.voisins(0)) - {1} == set(c.voisins(1)) - {0}
    assert c.frontiere(0, 1) is None and c.cercles(0) == c.cercles(1)
    tous = Carte(points([(1, 1)] * 8))
    verifier_tuiles(tous)
    assert tous.voisins(0) == list(range(1, 8)) and len(set(tous.coloration())) == 8


def test_coordonnees_negatives_grandes_et_bords():
    for positions, bornes in (([(-100, -200), (-500, -10), (-150, -20)], None),
                               ([(1e6, 1e6), (1e6 + 20, 1e6 + 10), (1e6 + 1, 1e6 + 15)], None),
                               ([(0, 0), (1000, 700), (1000, 0), (0, 700)], (0., 1000., 0., 700.))):
        c = Carte(points(positions), bornes)
        verifier_tuiles(c)
        assert all(c.plus_proche(x, y) == i for i, (x, y) in enumerate(positions))


def test_traversees_echantillonnage_fin():
    c = reelle()
    for debut, fin in (((0., 0.), (1000., 700.)), ((900., 90.), (30., 610.)), ((-100., 350.), (1100., 350.))):
        p, q = np.array(debut), np.array(fin)
        traversees = c.traversees(*debut, *fin)
        positions = p + np.linspace(0, 1, 30001)[:, None] * (q - p)
        indices = [c.plus_proche(float(x), float(y)) for x, y in positions]
        visites = [indices[0]] + [b for a, b in zip(indices[:-1], indices[1:]) if a != b]
        assert visites == [indices[0]] + [j for _, _, _, j in traversees]
        parametres = []
        for x, y, ancien, nouveau in traversees:
            t = np.dot(np.array((x, y)) - p, q - p) / np.dot(q - p, q - p)
            parametres.append(t)
            assert c.plus_proche(*(p + (t - 1e-8) * (q - p))) == ancien
            assert c.plus_proche(*(p + (t + 1e-8) * (q - p))) == nouveau
            assert math.isclose(c.distance(ancien, x, y), c.distance(nouveau, x, y), rel_tol=1e-10)
        assert parametres == sorted(parametres)
    assert c.traversees(430, 270, 430, 270) == []


def test_traversees_sommet_et_tangence():
    c = Carte(points([(-1, -1), (1, -1), (1, 1), (-1, 1)]))
    passages = c.traversees(-1, -1, 1, 1)
    assert len(passages) == 1 and np.allclose(passages[0][:2], (0, 0)) and passages[0][2:] == (0, 2)
    assert c.traversees(-1, -1, 0, 0) == []


def test_coloration_et_anneaux():
    for c in (reelle(), aleatoire(), Carte(points([(i, 0) for i in range(7)]))):
        couleurs = c.coloration()
        assert couleurs.shape == (c.n,) and np.all((0 <= couleurs) & (couleurs < 6))
        assert all(couleurs[i] != couleurs[j] for i, j in c.aretes)
        assert np.array_equal(couleurs, c.coloration())
    c = Carte(points([(i, 0) for i in range(5)]))
    assert c.voisins_ordre(0, 0) == {} and c.voisins_ordre(0, 2) == {1: 1, 2: 2}
    assert c.voisins_ordre(2, 1) == {1: 1, 3: 1}
    couleurs = c.coloration(1)
    assert all(couleurs[i] != couleurs[j] for i, j in c.aretes)


def test_csv_bom_point_virgule_et_options():
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "restaurants.csv"
        chemin.write_text(' I D ; Nom ; X ; y ;type;prix;note;nombre_avis\n'
                           '4;Le Café;-1;2;Italien;€€;"4,5";12\n8;Le Port;3;4;;3;;\n', encoding="utf-8-sig")
        r = charger_csv(chemin)
        assert r[0] == Restaurant(4, "Le Café", -1., 2., "Italien", 2, 4.5, 12)
        assert r[1].cuisine == "Autre" and r[1].prix == 3 and r[1].note is None and r[1].avis is None
        chemin.write_text('ID,nom,X,Y\n1,Simple,0,0\n', encoding="utf-8")
        assert charger_csv(chemin) == [Restaurant(1, "Simple", 0., 0.)]


def test_csv_erreurs_lisibles():
    cas = [('', None), ('ID,nom,X,Y\n', None), ('ID,nom,X\n1,A,2\n', '1'),
           ('ID,nom,X,Y\n1,A,nan,2\n', '2'), ('ID,nom,X,Y\n1,A,inf,2\n', '2'),
           ('ID,nom,X,Y\nabc,A,1,2\n', '2'), ('ID,nom,X,Y\n1,A,1\n', '2'),
           ('ID,nom,X,Y,note\n1,A,1,2,6\n', '2'), ('ID,nom,X,Y,prix\n1,A,1,2,€€€€€\n', '2'),
           ('ID,nom,X,Y,avis\n1,A,1,2,-1\n', '2'), ('ID,nom,X,Y\n1,A,1,2\n1,B,3,4\n', '3')]
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "restaurants.csv"
        for texte, ligne in cas:
            chemin.write_text(texte, encoding="utf-8")
            try:
                charger_csv(chemin)
            except ErreurDonnees as erreur:
                assert str(erreur)
                if ligne is not None:
                    assert f"ligne {ligne}" in str(erreur).casefold(), str(erreur)
            else:
                assert False, texte


def test_validation_et_generation():
    assert generer_restaurants(5, 4) == generer_restaurants(5, 4) and generer_restaurants(0) == []
    assert cadre([]) == (0., 1000., 0., 700.)
    for operation in (lambda: generer_restaurants(-1), lambda: generer_restaurants(1.5),
                      lambda: Carte(points([(0, 0)]), (1, 2, 1, 2)), lambda: Carte(points([(math.nan, 0)])),
                      lambda: Carte([], (0, 0, 1, 2)), lambda: reelle().plus_proche(math.inf, 1)):
        try:
            operation()
        except ErreurDonnees:
            pass
        else:
            assert False, operation
    try:
        reelle().voisins(-1)
    except IndexError:
        pass
    else:
        assert False, "Les indices négatifs doivent être refusés"


def test_performance_10000_restaurants():
    restaurants = generer_restaurants(10000, 19)
    debut = perf_counter()
    c = Carte(restaurants)
    duree = perf_counter() - debut
    assert duree < 1., f"Construction de 10000 sites : {duree:.3f} s"
    microsecondes = verifier_recherches(c, np.random.default_rng(5).uniform((0, 0), (1000, 700), (1000, 2)))
    assert microsecondes < 50 and set(c.temps_ms) == {"voronoi", "delaunay", "localisation"}
    print(f"  10000 sites : {duree * 1000:.1f} ms ; recherche : {microsecondes:.2f} µs")


if __name__ == "__main__":
    tests = [(nom, f) for nom, f in sorted(globals().items()) if nom.startswith("test_") and callable(f)]
    echecs, debut = 0, perf_counter()
    for nom, test in tests:
        try:
            test()
            print(f"OK  {nom}", flush=True)
        except Exception:
            echecs += 1
            print(f"ECHEC  {nom}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - echecs}/{len(tests)} tests réussis en {perf_counter() - debut:.2f} s.")
    sys.exit(bool(echecs))
