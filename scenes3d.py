"""Géométries pédagogiques de Voronoï/Delaunay, sans acteurs ni interface.

Les coordonnées VTK sont en double précision. Les identifiants ``site`` sont
les indices (à partir de zéro) des lignes du tableau fourni, pas ceux du CSV.
Les surfaces sont ouvertes et leurs triangles sont orientés vers le haut.
"""

from collections.abc import Callable
from operator import index

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def _sites(sites):
    sites = np.asarray(sites, dtype=float)
    if sites.ndim != 2 or sites.shape[1] != 2 or not len(sites):
        raise ValueError("sites doit être un tableau non vide de forme (n, 2)")
    if not np.isfinite(sites).all():
        raise ValueError("Les coordonnées des sites doivent être finies")
    return sites


def _cadre(bornes):
    cadre = np.asarray(bornes, dtype=float)
    if cadre.shape != (4,) or not np.isfinite(cadre).all():
        raise ValueError("bornes doit contenir xmin, xmax, ymin, ymax finis")
    if cadre[0] >= cadre[1] or cadre[2] >= cadre[3]:
        raise ValueError("Le cadre doit avoir une largeur et une hauteur positives")
    return cadre


def _positif(valeur, nom):
    if not np.isfinite(valeur) or valeur <= 0:
        raise ValueError(f"{nom} doit être strictement positif et fini")


def _entier(valeur, minimum, nom):
    try:
        valeur = index(valeur)
    except TypeError as erreur:
        raise ValueError(f"{nom} doit être entier") from erreur
    if valeur < minimum:
        raise ValueError(f"{nom} doit être au moins {minimum}")
    return valeur


def _tableau(donnees, nom):
    tableau = numpy_to_vtk(np.ascontiguousarray(donnees), deep=True)
    tableau.SetName(nom)
    return tableau


def _polydata(points, cellules, genre="polys"):
    poly = vtk.vtkPolyData()
    sommets = vtk.vtkPoints()
    sommets.SetData(_tableau(np.asarray(points, dtype=float).reshape(-1, 3), "Points"))
    poly.SetPoints(sommets)
    cellules = np.asarray(cellules, dtype=np.int64)
    if cellules.size:
        connectivite = vtk.vtkCellArray()
        connectivite.SetData(cellules.shape[1], _tableau(cellules.ravel(), "Indices"))
        {"polys": poly.SetPolys, "lines": poly.SetLines,
         "verts": poly.SetVerts}[genre](connectivite)
    return poly


def _normales(poly):
    if not poly.GetNumberOfPolys():
        return poly
    normales = vtk.vtkPolyDataNormals()
    normales.SetInputData(poly)
    # Les surfaces sont ouvertes : une orientation automatique serait ambiguë.
    normales.AutoOrientNormalsOff()
    normales.ConsistencyOff()
    normales.SplittingOff()
    normales.ComputeCellNormalsOn()
    normales.Update()
    return normales.GetOutput()


def _grille(cadre, nx, ny):
    x, y = np.meshgrid(np.linspace(cadre[0], cadre[1], nx),
                       np.linspace(cadre[2], cadre[3], ny))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(nx * ny)))
    bas = (np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)).ravel()
    faces = np.concatenate((np.column_stack((bas, bas + 1, bas + nx + 1)),
                            np.column_stack((bas, bas + nx + 1, bas + nx))))
    return points, faces


def _triangles(sites, triangles):
    triangles = np.asarray(triangles)
    if triangles.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if (triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)):
        raise ValueError("triangles doit contenir des indices entiers de forme (m, 3)")
    if triangles.min() < 0 or triangles.max() >= len(sites):
        raise ValueError("Indice de site hors limites")
    triangles = triangles.astype(np.int64, copy=True)
    a, b, c = sites[triangles].transpose(1, 0, 2)
    aire = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (
        b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    if np.any(aire == 0):
        raise ValueError("Un triangle est dégénéré dans le plan de la carte")
    triangles[aire < 0] = triangles[aire < 0, ::-1]
    return triangles


def cones(sites: np.ndarray, bornes, pente: float = 1.0,
          resolution: int = 96) -> vtk.vtkPolyData:
    """Construire un cône de distance par site, découpé au cadre vertical.

    ``resolution`` est le nombre minimal de secteurs angulaires (au moins 3).
    Un minimum de 4096 secteurs borne l'erreur radiale relative à 3e-7 : les
    intersections restent précises même près des frontières de Voronoï.
    Les cônes ne sont ni fusionnés ni bouchés ; chacun couvre tout le cadre.
    Les données entières ``site`` existent sur les points et sur les cellules.
    """
    sites, cadre = _sites(sites), _cadre(bornes)
    _positif(pente, "pente")
    # shortcut: approximation angulaire à 3e-7, augmenter resolution pour affiner.
    secteurs = max(4096, _entier(resolution, 3, "resolution"))
    angles = np.broadcast_to(np.arange(secteurs) * (2 * np.pi / secteurs),
                             (len(sites), secteurs))
    coins = cadre[[0, 2, 1, 2, 1, 3, 0, 3]].reshape(4, 2)
    interieur = ((sites > cadre[[0, 2]]) & (sites < cadre[[1, 3]])).all()
    if interieur:
        # Ajouter les coins évite qu'une corde ne tronque les angles du cadre.
        angles_coins = np.arctan2(coins[None, :, 1] - sites[:, None, 1],
                                 coins[None, :, 0] - sites[:, None, 0]) % (2 * np.pi)
        angles = np.sort(np.concatenate((angles, angles_coins), axis=1), axis=1)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=2)
    if interieur:
        limites = np.where(directions > 0, cadre[[1, 3]], cadre[[0, 2]])
        distances_plans = np.full(directions.shape, np.inf)
        np.divide(limites - sites[:, None, :], directions,
                  out=distances_plans, where=directions != 0)
        rayon = np.min(distances_plans, axis=2)
    else:
        # Les sites extérieurs demandent un vrai découpage des triangles.
        rayon = np.linalg.norm(coins[None, :, :] - sites[:, None, :], axis=2).max(axis=1)
        rayon = rayon[:, None] * (1.01 / np.cos(np.pi / secteurs))
    bord = sites[:, None, :] + directions * rayon[:, :, None]
    if interieur:
        bord = np.clip(bord, cadre[[0, 2]], cadre[[1, 3]])
    nombre = angles.shape[1]
    points = np.empty((len(sites), nombre + 1, 3))
    points[:, 0, :2] = sites
    points[:, 0, 2] = 0
    points[:, 1:, :2] = bord
    points[:, 1:, 2] = -pente * np.linalg.norm(bord - sites[:, None, :], axis=2)
    # Une direction passant exactement par un coin peut apparaître deux fois.
    valides = (np.roll(angles, -1, axis=1) - angles) % (2 * np.pi) > 1e-13
    tailles = valides.sum(axis=1)
    debuts = np.cumsum(np.r_[0, tailles[:-1] + 1])
    proprietaires = np.repeat(np.arange(len(sites), dtype=np.int32), tailles)
    courant = np.arange(tailles.sum()) + np.repeat(np.arange(len(sites)) + 1, tailles)
    suivant = courant + 1
    suivant[np.cumsum(tailles) - 1] = debuts + 1
    faces = np.column_stack((debuts[proprietaires], courant, suivant))
    points = points[np.column_stack((np.ones(len(sites), dtype=bool), valides))]
    poly = _polydata(points, faces)
    poly.GetPointData().SetScalars(_tableau(np.repeat(
        np.arange(len(sites), dtype=np.int32), tailles + 1), "site"))
    poly.GetCellData().SetScalars(_tableau(proprietaires, "site"))
    if not interieur:
        for origine, normale in (((cadre[0], 0, 0), (1, 0, 0)),
                                 ((cadre[1], 0, 0), (-1, 0, 0)),
                                 ((0, cadre[2], 0), (0, 1, 0)),
                                 ((0, cadre[3], 0), (0, -1, 0))):
            plan = vtk.vtkPlane()
            plan.SetOrigin(origine)
            plan.SetNormal(normale)
            decoupe = vtk.vtkClipPolyData()
            decoupe.SetInputData(poly)
            decoupe.SetClipFunction(plan)
            decoupe.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
            decoupe.Update()
            poly = decoupe.GetOutput()
        # Le découpeur interpole ses intersections : ramener l'arrondi au cadre.
        coordonnees = vtk_to_numpy(poly.GetPoints().GetData())
        coordonnees[:, :2] = np.clip(coordonnees[:, :2], cadre[[0, 2]], cadre[[1, 3]])
        poly.GetPoints().Modified()
        triangulation = vtk.vtkTriangleFilter()
        triangulation.SetInputData(poly)
        triangulation.Update()
        poly = triangulation.GetOutput()
        coordonnees = vtk_to_numpy(poly.GetPoints().GetData())
        faces = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        a, b, c = coordonnees[faces].transpose(1, 0, 2)
        valides = np.cross(b - a, c - a)[:, 2] > 0
        # Un sommet sur un plan peut engendrer un triangle d'aire nulle.
        utilises, indices = np.unique(faces[valides], return_inverse=True)
        ids_points = vtk_to_numpy(poly.GetPointData().GetArray("site"))[utilises]
        ids_faces = vtk_to_numpy(poly.GetCellData().GetArray("site"))[valides]
        coordonnees = coordonnees[utilises].copy()
        coordonnees[:, 2] = -pente * np.linalg.norm(
            coordonnees[:, :2] - sites[ids_points], axis=1)
        poly = _polydata(coordonnees, indices.reshape(-1, 3))
        poly.GetPointData().SetScalars(_tableau(ids_points, "site"))
        poly.GetCellData().SetScalars(_tableau(ids_faces, "site"))
    return _normales(poly)


def cretes(segments, sites: np.ndarray, pente: float = 1.0,
           echantillons: int = 32) -> vtk.vtkPolyData:
    """Échantillonner les frontières fournies sur l'enveloppe des cônes.

    Chaque élément est ``(début, fin, i, j)`` avec des indices de sites valides.
    ``echantillons`` est le nombre de points par polyligne, extrémités incluses.
    L'élévation couvre l'erreur des cordes, plus 1e-6 de l'échelle géométrique
    pour éviter le conflit de profondeur, y compris entre deux échantillons.
    Les tableaux entiers ``i`` et ``j`` sont des données de cellules.
    """
    sites = _sites(sites)
    _positif(pente, "pente")
    nombre = _entier(echantillons, 2, "echantillons")
    segments = list(segments)
    if not segments:
        poly = _polydata([], np.empty((0, nombre), dtype=np.int64), "lines")
        for nom in ("i", "j"):
            poly.GetCellData().AddArray(_tableau(np.empty(0, dtype=np.int32), nom))
        return poly
    extremites, paires = [], []
    for debut, fin, i, j in segments:
        extremites.append((debut, fin))
        paires.append((_entier(i, 0, "i"), _entier(j, 0, "j")))
    extremites = np.asarray(extremites, dtype=float)
    paires = np.asarray(paires, dtype=np.int32)
    if (extremites.shape != (len(segments), 2, 2)
            or not np.isfinite(extremites).all()):
        raise ValueError("Les extrémités doivent être des points 2D finis")
    if np.any(paires >= len(sites)) or np.any(paires[:, 0] == paires[:, 1]):
        raise ValueError("Chaque crête doit référencer deux sites distincts existants")
    demi_distance = np.linalg.norm(sites[paires[:, 0]] - sites[paires[:, 1]], axis=1) / 2
    if np.any(demi_distance == 0):
        raise ValueError("Deux sites confondus ne définissent pas de crête")
    xy = (extremites[:, :1, :] + np.linspace(0, 1, nombre)[None, :, None]
          * (extremites[:, 1:, :] - extremites[:, :1, :]))
    echelle = max(np.ptp(sites, axis=0).max(), np.ptp(extremites.reshape(-1, 2), axis=0).max())
    pas = np.linalg.norm(extremites[:, 1] - extremites[:, 0], axis=1) / (nombre - 1)
    # Sur une médiatrice, la courbure de la distance est bornée par 1/demi_distance.
    erreur_corde = np.minimum(pas / 2, pas ** 2 / (8 * demi_distance))
    epsilon = pente * (erreur_corde + max(echelle, np.finfo(float).eps) * 1e-6)
    z = (-pente * np.linalg.norm(xy - sites[paires[:, 0], None, :], axis=2)
         + epsilon[:, None])
    poly = _polydata(np.dstack((xy, z)).reshape(-1, 3),
                     np.arange(len(segments) * nombre).reshape(-1, nombre), "lines")
    for k, nom in enumerate(("i", "j")):
        poly.GetCellData().AddArray(_tableau(paires[:, k], nom))
    return poly


def enveloppe(sites: np.ndarray, bornes, pente: float = 1.0,
              resolution=(400, 280)) -> vtk.vtkPolyData:
    """Mailler le paysage ``z = -pente * distance au site le plus proche``.

    ``resolution=(nx, ny)`` compte les points, bords inclus (au moins 2 par axe).
    Les points portent ``site`` (entier, premier indice en cas d'égalité) et
    ``distance`` (avant multiplication par la pente). Les normales sont lissées.
    """
    sites, cadre = _sites(sites), _cadre(bornes)
    _positif(pente, "pente")
    if len(resolution) != 2:
        raise ValueError("resolution doit contenir nx et ny")
    nx, ny = (_entier(v, 2, "resolution") for v in resolution)
    points, faces = _grille(cadre, nx, ny)
    distances = np.full(len(points), np.inf)
    proches = np.zeros(len(points), dtype=np.int32)
    # Parcourir les sites évite une allocation grille × sites potentiellement énorme.
    for i, site in enumerate(sites):
        ecart = points[:, :2] - site
        carre = np.einsum("ij,ij->i", ecart, ecart)
        mieux = carre < distances
        distances[mieux] = carre[mieux]
        proches[mieux] = i
    distances = np.sqrt(distances)
    points[:, 2] = -pente * distances
    poly = _polydata(points, faces)
    poly.GetPointData().SetScalars(_tableau(proches, "site"))
    poly.GetPointData().AddArray(_tableau(distances, "distance"))
    return _normales(poly)


def normaliser(sites: np.ndarray, bornes) -> tuple[np.ndarray, Callable]:
    """Centrer le cadre et utiliser sa demi-dimension maximale comme échelle.

    Le cadre tient dans [-1, 1]² et les cercles restent ronds. L'inverse accepte
    tout tableau de coordonnées dont le dernier axe a deux composantes.
    """
    sites, cadre = _sites(sites), _cadre(bornes)
    centre = (cadre[[0, 2]] + cadre[[1, 3]]) / 2
    echelle = np.max(cadre[[1, 3]] - cadre[[0, 2]]) / 2

    def inverse(uv):
        """Revenir des coordonnées normalisées aux coordonnées de la carte."""
        uv = np.asarray(uv, dtype=float)
        if uv.ndim == 0 or uv.shape[-1] != 2 or not np.isfinite(uv).all():
            raise ValueError("Les coordonnées normalisées doivent être finies et 2D")
        return uv * echelle + centre

    return (sites - centre) / echelle, inverse


def paraboloide(sites: np.ndarray, triangles: np.ndarray, bornes,
                hauteur: float = 300.0, resolution: int = 80) -> dict[str, vtk.vtkPolyData]:
    """Relever les sites et les triangles fournis, en coordonnées de la carte.

    Renvoie ``surface``, ``points``, ``faces``, ``aretes``, ``tiges`` et ``plan``.
    La surface porte ``z`` ; les objets issus des sites portent ``site`` aux
    points. ``faces`` et ``plan`` portent aussi l'indice de cellule ``triangle``.
    Les normales des faces sont plates, celles de la surface sont lissées.
    Les sites situés dans le cadre sont insérés dans le maillage de la surface
    pour que leurs relèvements y appartiennent exactement.
    """
    sites, cadre = _sites(sites), _cadre(bornes)
    _positif(hauteur, "hauteur")
    nombre = _entier(resolution, 2, "resolution")
    triangles = _triangles(sites, triangles)
    uv, _ = normaliser(sites, cadre)
    releves = np.column_stack((sites, hauteur * np.einsum("ij,ij->i", uv, uv)))
    ids = np.arange(len(sites), dtype=np.int32)

    grille, _ = _grille(cadre, nombre, nombre)
    dedans = ((sites >= cadre[[0, 2]]) & (sites <= cadre[[1, 3]])).all(axis=1)
    grille = np.concatenate((grille, np.column_stack((sites[dedans], np.zeros(dedans.sum())))))
    triangulation = vtk.vtkDelaunay2D()
    triangulation.SetInputData(_polydata(grille, []))
    triangulation.SetTolerance(0.0)
    triangulation.Update()
    surface = vtk.vtkPolyData()
    surface.DeepCopy(triangulation.GetOutput())
    xy = vtk_to_numpy(surface.GetPoints().GetData())
    uv_grille, _ = normaliser(xy[:, :2], cadre)
    xy[:, 2] = hauteur * np.einsum("ij,ij->i", uv_grille, uv_grille)
    surface.GetPoints().Modified()
    surface.GetPointData().SetScalars(_tableau(xy[:, 2], "z"))

    points = _polydata(releves, ids[:, None], "verts")
    points.GetPointData().SetScalars(_tableau(ids, "site"))
    faces = _polydata(releves[triangles].reshape(-1, 3),
                      np.arange(triangles.size).reshape(-1, 3))
    faces.GetPointData().AddArray(_tableau(triangles.ravel().astype(np.int32), "site"))
    faces.GetCellData().SetScalars(_tableau(np.arange(len(triangles), dtype=np.int32), "triangle"))
    aretes_indices = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]],
        triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1), axis=0)
    aretes = _polydata(releves, aretes_indices, "lines")
    aretes.GetPointData().AddArray(_tableau(ids, "site"))
    for k, nom in enumerate(("i", "j")):
        aretes.GetCellData().AddArray(_tableau(aretes_indices[:, k].astype(np.int32), nom))
    bases = np.column_stack((sites, np.zeros(len(sites))))
    tiges = _polydata(np.stack((bases, releves), axis=1).reshape(-1, 3),
                      np.arange(2 * len(sites)).reshape(-1, 2), "lines")
    tiges.GetPointData().AddArray(_tableau(np.repeat(ids, 2), "site"))
    tiges.GetCellData().SetScalars(_tableau(ids, "site"))
    plan = _polydata(bases, triangles)
    plan.GetPointData().AddArray(_tableau(ids, "site"))
    plan.GetCellData().SetScalars(_tableau(np.arange(len(triangles), dtype=np.int32), "triangle"))
    return {"surface": _normales(surface), "points": points,
            "faces": _normales(faces), "aretes": aretes,
            "tiges": tiges, "plan": _normales(plan)}


def plan_triangle(sites: np.ndarray, tri, bornes,
                  hauteur: float = 300.0) -> tuple[float, float, float, float]:
    """Renvoyer le plan unitaire (a, b, c, d) relevé, avec c > 0.

    Les indices et l'orientation de ``tri`` sont vérifiés ; un triangle
    dégénéré est refusé. Les coefficients utilisent les coordonnées de la carte.
    """
    sites = _sites(sites)
    _positif(hauteur, "hauteur")
    triangle = _triangles(sites, [tri])
    uv, _ = normaliser(sites, bornes)
    points = np.column_stack((sites, hauteur * np.einsum("ij,ij->i", uv, uv)))[triangle[0]]
    normale = np.cross(points[1] - points[0], points[2] - points[0])
    normale /= np.linalg.norm(normale)
    return tuple(float(v) for v in (*normale, -np.dot(normale, points[0])))
