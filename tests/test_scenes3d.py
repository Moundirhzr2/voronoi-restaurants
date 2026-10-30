"""Tests autonomes : python -B tests/test_scenes3d.py [--captures]."""

import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from scenes3d import cones, cretes, enveloppe, normaliser, paraboloide, plan_triangle


BORNES = (0.0, 950.0, 0.0, 700.0)


def _donnees():
    with (RACINE / "data/restaurants.csv").open(encoding="utf-8-sig", newline="") as fichier:
        sites = np.array([[float(ligne["X"]), float(ligne["Y"])]
                          for ligne in csv.DictReader(fichier)])
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.column_stack((sites, np.zeros(len(sites)))), deep=True))
    entree = vtk.vtkPolyData()
    entree.SetPoints(points)
    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(entree)
    delaunay.SetTolerance(0.0)
    delaunay.Update()
    triangles = vtk_to_numpy(delaunay.GetOutput().GetPolys().GetConnectivityArray()).copy()
    return sites, triangles.reshape(-1, 3)


def _frontieres(sites):
    """Créer les segments de test par intersection indépendante de demi-plans."""
    segments = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            milieu = (sites[i] + sites[j]) / 2
            direction = (sites[j] - sites[i])[::-1] * (-1, 1)
            direction /= np.linalg.norm(direction)
            vecteurs = np.concatenate((np.eye(2), -np.eye(2), sites - sites[i]))
            limites = np.r_[BORNES[1] - milieu[0], BORNES[3] - milieu[1],
                            milieu[0] - BORNES[0], milieu[1] - BORNES[2],
                            np.sum((sites - sites[i]) ** 2, axis=1) / 2
                            - (sites - sites[i]) @ (milieu - sites[i])]
            coefficients = vecteurs @ direction
            paralleles = np.abs(coefficients) < 1e-10
            if np.any(limites[paralleles] < -1e-8):
                continue
            positifs, negatifs = coefficients > 1e-10, coefficients < -1e-10
            haut = np.min(limites[positifs] / coefficients[positifs], initial=np.inf)
            bas = np.max(limites[negatifs] / coefficients[negatifs], initial=-np.inf)
            if haut - bas > 1e-8:
                segments.append((milieu + bas * direction, milieu + haut * direction, i, j))
    return segments


def _points(poly):
    return vtk_to_numpy(poly.GetPoints().GetData())


def _faces(poly):
    assert poly.GetNumberOfCells() == poly.GetNumberOfPolys()
    assert all(poly.GetCellType(i) == vtk.VTK_TRIANGLE for i in range(poly.GetNumberOfCells()))
    return vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)


def _surface_propre(poly):
    points, faces = _points(poly), _faces(poly)
    assert points.dtype == np.float64
    assert np.isfinite(points).all()
    a, b, c = points[faces].transpose(1, 0, 2)
    normales = np.cross(b - a, c - a)
    assert np.all(normales[:, 2] > 0), "Triangle dégénéré ou orienté vers le bas"
    for attributs in (poly.GetPointData(), poly.GetCellData()):
        valeurs = vtk_to_numpy(attributs.GetNormals())
        assert np.isfinite(valeurs).all()
        assert np.allclose(np.linalg.norm(valeurs, axis=1), 1, atol=2e-6)
        assert np.all(valeurs[:, 2] > 0)


def test_cones_cadre_donnees_et_normales():
    sites, _ = _donnees()
    poly = cones(sites, BORNES, pente=1.25)
    _surface_propre(poly)
    assert np.array_equal(np.array(poly.GetBounds())[:4], BORNES)
    points, faces = _points(poly), _faces(poly)
    ids = vtk_to_numpy(poly.GetPointData().GetArray("site"))
    cellules = vtk_to_numpy(poly.GetCellData().GetArray("site"))
    assert np.issubdtype(ids.dtype, np.integer)
    assert np.issubdtype(cellules.dtype, np.integer)
    assert np.array_equal(ids[faces], np.repeat(cellules[:, None], 3, axis=1))
    assert np.allclose(points[:, 2], -1.25 * np.linalg.norm(points[:, :2] - sites[ids], axis=1))
    a, b, c = points[faces].transpose(1, 0, 2)
    aires = np.cross(b - a, c - a)[:, 2] / 2
    assert np.allclose(np.bincount(cellules, weights=aires),
                       (BORNES[1] - BORNES[0]) * (BORNES[3] - BORNES[2]), rtol=1e-12)


def test_cones_5000_rayons():
    sites, _ = _donnees()
    poly = cones(sites, BORNES)
    locator = vtk.vtkCellLocator()
    # Les triangles en éventail ont de grandes boîtes : limiter les octants
    # évite de les dupliquer dans des milliers de compartiments inutiles.
    locator.AutomaticOff()
    locator.SetMaxLevel(2)
    locator.SetDataSet(poly)
    locator.BuildLocator()
    candidats = np.random.default_rng(20261030).uniform((0, 0), (950, 700), (10000, 2))
    carres = np.sum((candidats[:, None, :] - sites[None, :, :]) ** 2, axis=2)
    ordre = np.argsort(carres, axis=1)
    i, j = ordre[:, 0], ordre[:, 1]
    marge = ((carres[np.arange(len(candidats)), j] - carres[np.arange(len(candidats)), i])
             / (2 * np.linalg.norm(sites[j] - sites[i], axis=1)))
    echelle = np.hypot(BORNES[1] - BORNES[0], BORNES[3] - BORNES[2])
    gardes = np.flatnonzero(marge > 1e-6 * echelle)[:5000]
    assert len(gardes) == 5000
    temps, sous_id, cellule_id = vtk.reference(0.0), vtk.reference(0), vtk.reference(0)
    intersection, locales, cellule = [0.0] * 3, [0.0] * 3, vtk.vtkGenericCell()
    labels = vtk_to_numpy(poly.GetCellData().GetArray("site"))
    for k in gardes:
        x, y = candidats[k]
        touche = locator.IntersectWithLine((x, y, echelle), (x, y, -2 * echelle),
            echelle * 1e-12, temps, intersection, locales, sous_id, cellule_id, cellule)
        assert touche, f"Rayon sans intersection en {x, y}"
        assert labels[int(cellule_id)] == i[k], f"Mauvais site en {x, y}"


def test_cones_sites_exterieurs_et_sur_bord():
    sites = np.array([[-100., 350.], [0., 0.], [500., 350.]])
    poly = cones(sites, BORNES, resolution=48)
    _surface_propre(poly)
    assert np.array_equal(np.array(poly.GetBounds())[:4], BORNES)
    points = _points(poly)
    ids = vtk_to_numpy(poly.GetPointData().GetArray("site"))
    assert np.allclose(points[:, 2], -np.linalg.norm(points[:, :2] - sites[ids], axis=1),
                       atol=1e-3, rtol=1e-6)


def test_cretes_equidistance_et_enveloppe():
    sites, _ = _donnees()
    segments = _frontieres(sites)
    assert len(segments) > 50
    poly = cretes(segments, sites, pente=1.7, echantillons=32)
    assert poly.GetNumberOfLines() == len(segments)
    points = _points(poly).reshape(-1, 32, 3)
    i = vtk_to_numpy(poly.GetCellData().GetArray("i"))
    j = vtk_to_numpy(poly.GetCellData().GetArray("j"))
    di = np.linalg.norm(points[:, :, :2] - sites[i, None, :], axis=2)
    dj = np.linalg.norm(points[:, :, :2] - sites[j, None, :], axis=2)
    assert np.allclose(di, dj, atol=1e-9, rtol=1e-12)
    proches = np.linalg.norm(points[:, :, None, :2] - sites, axis=3).min(axis=2)
    assert np.all(points[:, :, 2] >= -1.7 * proches)
    assert np.max(points[:, :, 2] + 1.7 * proches) < 1
    for fraction in np.linspace(0, 1, 9):
        cordes = points[:, :-1] * (1 - fraction) + points[:, 1:] * fraction
        distances = np.linalg.norm(cordes[:, :, None, :2] - sites, axis=3).min(axis=2)
        assert np.all(cordes[:, :, 2] >= -1.7 * distances)
    assert np.allclose(points[:, 0, :2], [s[0] for s in segments])
    assert np.allclose(points[:, -1, :2], [s[1] for s in segments])
    vide = cretes([], sites)
    assert vide.GetNumberOfPoints() == vide.GetNumberOfCells() == 0
    assert vide.GetCellData().GetArray("i") is not None
    assert vide.GetCellData().GetArray("j") is not None


def test_enveloppe_chaque_point():
    sites, _ = _donnees()
    poly = enveloppe(sites, BORNES, pente=0.6)
    _surface_propre(poly)
    assert poly.GetNumberOfPoints() == 400 * 280
    assert np.array_equal(np.array(poly.GetBounds())[:4], BORNES)
    points = _points(poly)
    distances = np.linalg.norm(points[:, None, :2] - sites, axis=2)
    assert np.array_equal(vtk_to_numpy(poly.GetPointData().GetArray("site")),
                          np.argmin(distances, axis=1))
    assert np.allclose(vtk_to_numpy(poly.GetPointData().GetArray("distance")),
                       distances.min(axis=1))
    assert np.allclose(points[:, 2], -0.6 * distances.min(axis=1))


def test_normalisation_isotrope_et_inverse():
    sites, _ = _donnees()
    copie = sites.copy()
    uv, inverse = normaliser(sites, BORNES)
    assert np.max(np.abs(uv)) <= 1
    assert np.allclose(inverse(uv), sites)
    assert np.array_equal(sites, copie)
    distances = np.linalg.norm(sites[1:] - sites[0], axis=1)
    assert np.allclose(np.linalg.norm(uv[1:] - uv[0], axis=1) / distances, 2 / 950)
    assert np.allclose(inverse([0, 0]), [475, 350])


def test_paraboloide_enveloppe_convexe_inferieure():
    sites, triangles = _donnees()
    copie = triangles.copy()
    scene = paraboloide(sites, triangles, BORNES)
    assert set(scene) == {"surface", "points", "faces", "aretes", "tiges", "plan"}
    assert np.array_equal(triangles, copie)
    for nom in ("surface", "faces", "plan"):
        _surface_propre(scene[nom])
    releves = _points(scene["points"])
    uv, _ = normaliser(sites, BORNES)
    assert np.allclose(releves[:, 2], 300 * np.sum(uv ** 2, axis=1))
    assert np.array_equal(releves[:, :2], sites)
    assert scene["points"].GetNumberOfVerts() == len(sites)
    assert scene["faces"].GetNumberOfPolys() == len(triangles)
    for triangle in triangles:
        a, b, c, d = plan_triangle(sites, triangle, BORNES)
        assert c > 0
        z_plan = -(a * releves[:, 0] + b * releves[:, 1] + d) / c
        tolerance = 1e-9 * max(1, np.max(np.abs(z_plan)), np.max(np.abs(releves[:, 2])))
        assert np.all(releves[:, 2] - z_plan >= -tolerance)
        assert np.allclose(releves[triangle, 2], z_plan[triangle], atol=tolerance, rtol=0)
        assert np.allclose(plan_triangle(sites, triangle[::-1], BORNES), (a, b, c, d))
    # Des normales identiques aux trois sommets conservent l'aspect plat des faces.
    normales = vtk_to_numpy(scene["faces"].GetPointData().GetNormals()).reshape(-1, 3, 3)
    assert np.allclose(normales, normales[:, :1, :])


def test_paraboloide_sites_sur_surface_et_connectivite():
    sites, triangles = _donnees()
    scene = paraboloide(sites, triangles, BORNES, hauteur=170, resolution=30)
    surface = scene["surface"]
    points = _points(surface)
    uv, _ = normaliser(points[:, :2], BORNES)
    assert np.allclose(points[:, 2], 170 * np.sum(uv ** 2, axis=1))
    assert np.array_equal(vtk_to_numpy(surface.GetPointData().GetArray("z")), points[:, 2])
    assert np.array_equal(np.array(surface.GetBounds())[:4], BORNES)
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    trouve, cellule_id, sous_id, carre = [0.] * 3, vtk.reference(0), vtk.reference(0), vtk.reference(0.)
    for point in _points(scene["points"]):
        locator.FindClosestPoint(point, trouve, cellule_id, sous_id, carre)
        assert float(carre) < 1e-18
    aretes = vtk_to_numpy(scene["aretes"].GetLines().GetConnectivityArray()).reshape(-1, 2)
    attendues = {tuple(sorted((int(t[i]), int(t[j]))))
                 for t in triangles for i, j in ((0, 1), (1, 2), (2, 0))}
    assert {tuple(a) for a in aretes} == attendues
    assert len(aretes) == len(attendues)
    tiges = _points(scene["tiges"]).reshape(-1, 2, 3)
    assert np.array_equal(tiges[:, 0, :2], sites)
    assert np.all(tiges[:, 0, 2] == 0)
    assert np.array_equal(tiges[:, 1, :], _points(scene["points"]))
    assert np.all(_points(scene["plan"])[:, 2] == 0)


def test_petits_ensembles_et_egalites():
    sites = np.array([[2., 3.], [2., 3.]])
    poly = enveloppe(sites, (0, 10, 0, 8), resolution=(2, 3))
    assert np.all(vtk_to_numpy(poly.GetPointData().GetArray("site")) == 0)
    scene = paraboloide(sites[:1], np.empty((0, 3), dtype=int), (0, 10, 0, 8), resolution=4)
    assert scene["faces"].GetNumberOfCells() == 0
    assert scene["aretes"].GetNumberOfCells() == 0
    assert scene["points"].GetNumberOfVerts() == 1


def test_entrees_invalides():
    sites = np.array([[1., 1.], [2., 1.], [1., 2.]])
    cadre = (0, 3, 0, 3)
    operations = (
        lambda: cones([], cadre),
        lambda: cones([[np.nan, 1]], cadre),
        lambda: cones(sites, (0, 0, 0, 3)),
        lambda: cones(sites, cadre, pente=0),
        lambda: cones(sites, cadre, resolution=2.5),
        lambda: cretes([((0, 0), (1, 1), 0, 3)], sites),
        lambda: cretes([((0, 0), (1, 1), 0, 0)], sites),
        lambda: cretes([], sites, echantillons=1),
        lambda: enveloppe(sites, cadre, resolution=(1, 5)),
        lambda: normaliser(sites, (0, np.inf, 0, 1)),
        lambda: paraboloide(sites, [[0, 1, 3]], cadre),
        lambda: paraboloide(sites, [[0., 1., 2.]], cadre),
        lambda: paraboloide(sites, [[0, 1, 2]], cadre, hauteur=-1),
        lambda: plan_triangle(sites, [0, 0, 1], cadre),
        lambda: plan_triangle(np.array([[0., 0.], [1., 1.], [2., 2.]]), [0, 1, 2], cadre),
    )
    for operation in operations:
        try:
            operation()
        except ValueError:
            pass
        else:
            assert False, "Entrée invalide acceptée"


def test_performances():
    sites, _ = _donnees()
    for points, resolution, limite in ((sites, 96, 0.2),
            (np.random.default_rng(48).uniform((0, 0), (950, 700), (1000, 2)), 48, 3.0)):
        debut = perf_counter()
        poly = cones(points, BORNES, resolution=resolution)
        duree = perf_counter() - debut
        assert poly.GetNumberOfCells() > 0
        print(f"  cones({len(points)}, resolution={resolution}) : {duree:.4f} s < {limite} s", flush=True)
        assert duree < limite, f"Construction trop lente : {duree:.4f} s"
        del poly


def _captures():
    """Produire les trois contrôles hors écran sans ajouter d'acteurs au module."""
    sites, triangles = _donnees()
    repertoire = RACINE / "captures/codex"
    repertoire.mkdir(parents=True, exist_ok=True)
    palette = vtk.vtkLookupTable()
    palette.SetNumberOfTableValues(len(sites))
    palette.SetTableRange(0, len(sites) - 1)
    couleurs = vtk.vtkColorSeries()
    couleurs.SetColorScheme(vtk.vtkColorSeries.BREWER_QUALITATIVE_SET3)
    for i in range(len(sites)):
        couleur = couleurs.GetColor(i % couleurs.GetNumberOfColors())
        facteur = (1.0, 0.78, 0.60)[i // couleurs.GetNumberOfColors()]
        palette.SetTableValue(i, *(facteur * c / 255 for c in couleur), 1)
    palette.Build()

    def acteur(poly, couleur, largeur=1, sites_colores=False, opacite=1):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()
        if sites_colores:
            mapper.ScalarVisibilityOn()
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray("site")
            mapper.SetLookupTable(palette)
            mapper.SetScalarRange(0, len(sites) - 1)
        objet = vtk.vtkActor()
        objet.SetMapper(mapper)
        objet.GetProperty().SetColor(couleur)
        objet.GetProperty().SetLineWidth(largeur)
        objet.GetProperty().SetOpacity(opacite)
        objet.GetProperty().SetPointSize(9)
        return objet

    def sauver(nom, objets, titre, dessus=False):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.96, 0.965, 0.98)
        for objet in objets:
            renderer.AddActor(objet)
        texte = vtk.vtkTextActor()
        texte.SetInput(titre)
        texte.SetPosition(35, 750)
        texte.GetTextProperty().SetFontSize(25)
        texte.GetTextProperty().SetColor(0.12, 0.16, 0.22)
        renderer.AddViewProp(texte)
        fenetre = vtk.vtkRenderWindow()
        fenetre.SetOffScreenRendering(1)
        fenetre.SetSize(1200, 800)
        fenetre.SetMultiSamples(0)
        fenetre.AddRenderer(renderer)
        camera = renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetFocalPoint(475, 350, -100 if nom.startswith("cones") else 100)
        camera.SetPosition(475, 350, 2500) if dessus else camera.SetPosition(1800, -1500, 1500)
        camera.SetViewUp(0, 1, 0) if dessus else camera.SetViewUp(0, 0, 1)
        renderer.ResetCamera()
        if dessus:
            camera.SetParallelScale(410)
        renderer.ResetCameraClippingRange()
        fenetre.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(fenetre)
        capture.ReadFrontBufferOff()
        capture.Update()
        assert capture.GetOutput().GetDimensions() == (1200, 800, 1)
        pixels = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
        assert np.ptp(pixels) > 100
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(repertoire / nom))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        fenetre.Finalize()
        print(f"  capture : {repertoire / nom}", flush=True)

    poly = cones(sites, BORNES, pente=0.6)
    cone_acteur = acteur(poly, (1, 1, 1), sites_colores=True)
    cone_acteur.GetProperty().LightingOff()
    lignes = acteur(cretes(_frontieres(sites), sites, pente=0.6), (0.14, 0.17, 0.22), 2)
    # L'épaisseur raster déborde des crêtes : avancer leur profondeur de rendu.
    lignes.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
    lignes.GetMapper().SetRelativeCoincidentTopologyLineOffsetParameters(0, -5000)
    points = vtk.vtkPolyData()
    sommets = vtk.vtkPoints()
    sommets.SetData(numpy_to_vtk(np.column_stack((sites, np.full(len(sites), 2.0))), deep=True))
    points.SetPoints(sommets)
    sommets_filtre = vtk.vtkVertexGlyphFilter()
    sommets_filtre.SetInputData(points)
    sommets_filtre.Update()
    points_acteur = acteur(sommets_filtre.GetOutput(), (0.08, 0.10, 0.15))
    sauver("cones_dessus.png", [cone_acteur, lignes, points_acteur],
           "Cones de distance - vue orthographique de dessus", dessus=True)
    cone_acteur.GetProperty().LightingOn()
    sauver("cones_oblique.png", [cone_acteur, lignes, points_acteur],
           "Cones de distance - la surface visible donne Voronoi")
    scene = paraboloide(sites, triangles, BORNES)
    surface = acteur(scene["surface"], (0.44, 0.70, 0.88), opacite=0.24)
    faces = acteur(scene["faces"], (0.95, 0.64, 0.26), opacite=0.75)
    surface.GetProperty().LightingOff()
    faces.GetProperty().LightingOff()
    projection = acteur(scene["plan"], (0.34, 0.45, 0.55), 2)
    projection.GetProperty().SetRepresentationToWireframe()
    sauver("paraboloide_oblique.png", [surface, faces,
        acteur(scene["aretes"], (0.55, 0.25, 0.09), 2),
        acteur(scene["tiges"], (0.45, 0.47, 0.52), 1.5),
        acteur(scene["points"], (0.65, 0.12, 0.12)), projection],
        "Relevement sur le paraboloide - faces inferieures et projection")


if __name__ == "__main__":
    tests = [(nom, fonction) for nom, fonction in list(globals().items()) if nom.startswith("test_")]
    debut = perf_counter()
    for nom, fonction in tests:
        instant = perf_counter()
        fonction()
        print(f"OK {nom} ({perf_counter() - instant:.3f} s)", flush=True)
    print(f"{len(tests)} tests réussis en {perf_counter() - debut:.3f} s", flush=True)
    if "--captures" in sys.argv:
        _captures()
