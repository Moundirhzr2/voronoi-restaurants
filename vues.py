"""Rendu VTK de l'application : la carte 2D et les deux vues 3D pédagogiques.

Ce module ne dépend pas de Qt. L'interface lui donne une `geometrie.Carte` et appelle ses
méthodes ; chaque animation passe par une fonction qui reçoit t dans [0, 1], l'horloge
restant côté interface.
"""
import math
import unicodedata
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import scenes3d

# Jetons du thème sombre : fond presque noir, un seul accent lavande.
JETONS = {
    "canvas": "#010102", "s1": "#0f1011", "s2": "#141516", "s3": "#18191a", "s4": "#191a1b",
    "hairline": "#23252a", "hairline2": "#34343a", "hairline3": "#3e3e44",
    "ink": "#f7f8f8", "ink2": "#d0d6e0", "ink3": "#8a8f98", "ink4": "#62666d",
    "accent": "#5e6ad2", "accent2": "#828fff", "succes": "#27a644",
}

# Familles de cuisine : palette catégorielle sombre du skill dataviz, sans le violet
# (réservé à l'accent de l'interface). Testée : sur une carte toutes les paires se touchent,
# donc la couleur ne porte jamais seule l'identité (noms, infobulles, légende).
FAMILLES = [
    ("Français", "#3987e5", ("francais", "bistrot", "brasserie")),
    ("Italien", "#e66767", ("italien", "pizzeria", "pizza")),
    ("Asiatique", "#d55181", ("japonais", "asiatique", "chinois", "thai", "vietnamien", "coreen")),
    ("Indien et oriental", "#c98500", ("indien", "oriental", "libanais", "marocain", "turc")),
    ("Végétarien", "#008300", ("vegetarien", "vegan", "vegetalien")),
    ("Fruits de mer", "#199e70", ("fruits de mer", "poisson")),
    ("Grill et mexicain", "#d95926", ("burgers", "burger", "grillades", "grill", "mexicain", "tex-mex")),
]
AUTRE = ("Autres", "#8a8f98")

# Mosaïque neutre : deux tuiles voisines n'ont jamais le même gris (coloration du graphe).
MOSAIQUE = ["#121318", "#181a20", "#1e2028", "#24272f", "#2a2d36", "#30333d"]
# Dégradé séquentiel bleu (une seule teinte) pour la note : clair = mieux noté.
RAMPE_NOTE = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
              "#5598e7", "#6da7ec", "#86b6ef"]

_POLICES = Path("C:/Windows/Fonts")
POLICE = _POLICES / "segoeui.ttf"
POLICE_GRASSE = _POLICES / "seguisb.ttf"


def rgb(couleur):
    """'#rrggbb' -> (r, g, b) dans [0, 1] ; laisse passer un tuple."""
    if not isinstance(couleur, str):
        return tuple(couleur)
    c = couleur.lstrip("#")
    return tuple(int(c[k:k + 2], 16) / 255 for k in (0, 2, 4))


def melange(a, b, part_b):
    a, b = rgb(a), rgb(b)
    return tuple(x * (1 - part_b) + y * part_b for x, y in zip(a, b))


def _sans_accents(texte):
    texte = unicodedata.normalize("NFD", texte.lower().strip())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def famille(cuisine):
    """(nom de famille, couleur) d'une cuisine ; « Autres » si elle n'est pas connue."""
    c = _sans_accents(cuisine or "")
    for nom, couleur, mots in FAMILLES:
        if c in mots:
            return nom, couleur
    return AUTRE


def style_texte(prop, taille, couleur="ink2", gras=False):
    prop.SetFontSize(taille)
    prop.SetColor(rgb(JETONS.get(couleur, couleur)))
    police = POLICE_GRASSE if gras else POLICE
    if police.exists():  # sinon, police intégrée à VTK
        prop.SetFontFamily(vtk.VTK_FONT_FILE)
        prop.SetFontFile(str(police))


# --- Petites fabriques de géométrie -------------------------------------------------

def poly_segments(segments, z=0.0):
    points, lignes = vtk.vtkPoints(), vtk.vtkCellArray()
    for (x0, y0), (x1, y1) in segments:
        a = points.InsertNextPoint(x0, y0, z)
        b = points.InsertNextPoint(x1, y1, z)
        lignes.InsertNextCell(2, (a, b))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lignes)
    return poly


def poly_lignes(polylignes, z=0.0, fermees=False):
    """Plusieurs polylignes [(k, 2) ou (k, 3)] dans un seul polydata."""
    points, lignes = vtk.vtkPoints(), vtk.vtkCellArray()
    for coords in polylignes:
        ids = []
        for p in coords:
            ids.append(points.InsertNextPoint(p[0], p[1], p[2] if len(p) > 2 else z))
        if fermees and ids:
            ids.append(ids[0])
        lignes.InsertNextCell(len(ids), ids)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lignes)
    return poly


def poly_polygones(polygones, z=0.0):
    points, cellules = vtk.vtkPoints(), vtk.vtkCellArray()
    for coords in polygones:
        cellules.InsertNextCell(len(coords), [points.InsertNextPoint(x, y, z) for x, y in coords])
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cellules)
    return poly


def poly_points(coords, z=0.0, couleurs=None):
    """Nuage de points avec sommets (verts) et, si fournies, couleurs RGB 0-255 par point."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    xyz = np.column_stack([coords, np.full(len(coords), z)])
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))
    verts = vtk.vtkCellArray()
    for k in range(len(coords)):
        verts.InsertNextCell(1, (k,))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(verts)
    if couleurs is not None:
        tableau = numpy_to_vtk(np.asarray(couleurs, dtype=np.uint8), deep=True)
        tableau.SetName("rgb")
        poly.GetPointData().SetScalars(tableau)
    return poly


def cercle(cx, cy, r, n=128):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def pointilles(p, q, tiret, vide):
    """Découpe le segment p -> q en tirets (le pointillé OpenGL n'existe plus en VTK 9)."""
    (x0, y0), (x1, y1) = p, q
    longueur = math.hypot(x1 - x0, y1 - y0)
    if longueur == 0:
        return []
    ux, uy = (x1 - x0) / longueur, (y1 - y0) / longueur
    morceaux, s = [], 0.0
    while s < longueur:
        e = min(s + tiret, longueur)
        morceaux.append(((x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)))
        s = e + vide
    return morceaux


def couleurs_uint8(couleurs):
    return (np.asarray([rgb(c) for c in couleurs], dtype=float).reshape(-1, 3) * 255).round().astype(np.uint8)


def acteur(poly=None, couleur="ink", opacite=1.0, largeur=1.0, z=0.0, eclairage=False):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly if poly is not None else vtk.vtkPolyData())
    mapper.ScalarVisibilityOff()  # couleur unie, sauf si un acteur active ses propres couleurs
    a = vtk.vtkActor()
    a.SetMapper(mapper)
    p = a.GetProperty()
    p.SetColor(rgb(JETONS.get(couleur, couleur) if isinstance(couleur, str) else couleur))
    p.SetOpacity(opacite)
    p.SetLineWidth(largeur)
    if not eclairage:
        p.LightingOff()  # couleurs plates : c'est une carte, pas une scène éclairée
    a.SetPosition(0, 0, z)
    return a


def remplacer(a, poly):
    a.GetMapper().SetInputData(poly)


# --- Carte 2D --------------------------------------------------------------------------

class Carte2D:
    """Carte 2D : tuiles de Voronoï, triangulation de Delaunay, marqueurs, sélection animée."""

    MODES = ("mosaique", "cuisine", "note")

    def __init__(self):
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(rgb(JETONS["canvas"]))
        self.ren.GetActiveCamera().ParallelProjectionOn()
        self.carte = None
        self.cadre = (0, 1, 0, 1)
        self.mode = "mosaique"
        self.taille_px = (800, 600)
        self.echelle_ajustee = 1.0
        self.selection_courante = None

        # Couches, de bas en haut (z en unités de la carte, caméra orthographique vers -z).
        # Les points rendus en sphères « gonflent » vers la caméra d'environ leur rayon :
        # les couches de points sont donc très espacées pour ne pas se masquer entre elles.
        self.a_tuiles = acteur(z=0.0)
        self.a_voisines = acteur(couleur="ink", opacite=0.05, z=1.0)
        self.a_selection = acteur(z=1.2)
        self.a_survol = acteur(couleur="ink", opacite=0.07, z=1.4)
        self.a_bords = acteur(couleur="#30333c", largeur=1.0, z=2.0)
        self.a_delaunay = acteur(couleur="ink3", opacite=0.55, largeur=1.0, z=3.0)
        self.a_survol_bord = acteur(couleur="ink4", largeur=1.5, z=4.0)
        self.a_halo = acteur(couleur="accent", opacite=0.25, largeur=8.0, z=5.0)
        self.a_contour = acteur(couleur="accent2", largeur=2.5, z=5.5)
        self.a_fleur = acteur(couleur="accent2", opacite=0.38, largeur=1.2, z=6.0)
        self.a_dual_vor = acteur(couleur="accent2", largeur=5.0, z=7.0)
        self.a_dual_del = acteur(couleur="ink", largeur=1.6, z=7.5)
        self.a_liens = acteur(couleur="accent2", opacite=0.9, largeur=2.0, z=8.0)
        self.a_lien_user = acteur(couleur="ink2", largeur=1.6, z=8.5)
        self.a_trajet = acteur(couleur="ink3", opacite=0.75, largeur=1.6, z=9.0)
        self.a_sommets = acteur(couleur="accent2", z=12.0)
        self.a_sommets.GetProperty().SetPointSize(5)
        self.a_croisements = acteur(couleur="ink", z=14.0)
        self.a_croisements.GetProperty().SetPointSize(7)
        # Marqueurs en pixels (taille fixe à l'écran, même en zoomant) : anneau sombre + pastille.
        self.a_marq_voisins = acteur(couleur="accent2", z=20.0)
        self.a_marq_voisins.GetProperty().SetPointSize(16)
        self.a_marq_fond = acteur(couleur="canvas", z=40.0)
        self.a_marq_fond.GetProperty().SetPointSize(12)
        self.a_marq = acteur(z=60.0)
        self.a_marq.GetProperty().SetPointSize(8)
        self.a_pulse = acteur(couleur="accent2", largeur=2.0, z=70.0)
        self.a_user_anneau = acteur(couleur="accent", z=80.0)
        self.a_user_anneau.GetProperty().SetPointSize(22)
        self.a_user = acteur(couleur="ink", z=110.0)
        self.a_user.GetProperty().SetPointSize(11)
        for a in (self.a_sommets, self.a_croisements, self.a_marq_fond, self.a_marq, self.a_marq_voisins,
                  self.a_user_anneau, self.a_user):
            a.GetProperty().RenderPointsAsSpheresOn()

        self.a_distance = vtk.vtkTextActor()
        style_texte(self.a_distance.GetTextProperty(), 13, "ink", gras=True)
        tp = self.a_distance.GetTextProperty()
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToCentered()
        tp.SetBackgroundColor(rgb(JETONS["s3"]))
        tp.SetBackgroundOpacity(0.92)
        tp.FrameOn()
        tp.SetFrameColor(rgb(JETONS["hairline3"]))
        self.a_distance.GetPositionCoordinate().SetCoordinateSystemToWorld()
        self.a_distance.VisibilityOff()

        # Noms : placement sans chevauchement calculé ici à chaque changement de caméra
        # (voir _etiquettes), puis simple affichage, très rapide, à chaque image.
        self.pts_noms = vtk.vtkPolyData()
        self.mapper_noms = vtk.vtkLabeledDataMapper()
        self.mapper_noms.SetInputData(self.pts_noms)
        self.mapper_noms.SetLabelModeToLabelFieldData()
        self.mapper_noms.SetFieldDataName("nom")
        tp = self.mapper_noms.GetLabelTextProperty()
        style_texte(tp, 12, "ink2")
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToCentered()
        tp.ShadowOn()  # ombre portée sombre : lisible sur toutes les teintes de cellule
        tp.SetShadowOffset(1, -1)
        self.a_noms = vtk.vtkActor2D()
        self.a_noms.SetMapper(self.mapper_noms)

        self.calques = {"voronoi": [self.a_tuiles, self.a_bords], "delaunay": [self.a_delaunay],
                        "noms": [self.a_noms], "fleur": [self.a_fleur, self.a_sommets],
                        "trajet": [self.a_trajet, self.a_croisements]}
        self.a_delaunay.VisibilityOff()
        for a in (self.a_tuiles, self.a_voisines, self.a_selection, self.a_survol, self.a_bords,
                  self.a_delaunay, self.a_survol_bord, self.a_halo, self.a_contour, self.a_fleur,
                  self.a_sommets, self.a_dual_vor, self.a_dual_del, self.a_liens, self.a_lien_user,
                  self.a_trajet, self.a_croisements, self.a_marq_voisins, self.a_marq_fond, self.a_marq,
                  self.a_pulse, self.a_user_anneau, self.a_user, self.a_distance, self.a_noms):
            self.ren.AddViewProp(a)

    # -- données -----------------------------------------------------------------------

    def charger(self, carte, cadre):
        """Reconstruit les couches de base pour une nouvelle Carte (après un filtre par exemple)."""
        self.carte, self.cadre = carte, cadre
        self.effacer_selection()
        self.survol(None)
        self.trajet_effacer()
        n = carte.n
        self.couleurs_famille = couleurs_uint8([famille(r.cuisine)[1] for r in carte.restaurants])
        self.coloration = carte.coloration(len(MOSAIQUE)) if n else np.zeros(0, dtype=int)

        bords = vtk.vtkExtractEdges()
        bords.SetInputData(carte.voronoi)
        bords.Update()
        remplacer(self.a_bords, bords.GetOutput())
        # Arêtes plutôt que triangles : couvre aussi les points alignés (chaîne sans triangle).
        remplacer(self.a_delaunay, poly_segments([(carte.sites[i], carte.sites[j]) for i, j in carte.aretes]))

        # Pastilles plus petites quand la carte est dense, sinon elles cachent les cellules.
        taille = 8 if n <= 200 else 5 if n <= 2000 else 3
        self.a_marq.GetProperty().SetPointSize(taille)
        self.a_marq_fond.GetProperty().SetPointSize(taille + 4)
        self.a_marq_fond.SetVisibility(n <= 2000)
        remplacer(self.a_marq_fond, poly_points(carte.sites))
        remplacer(self.a_marq, poly_points(carte.sites, couleurs=self.couleurs_famille))
        self.a_marq.GetMapper().SetColorModeToDirectScalars()
        self.a_marq.GetMapper().ScalarVisibilityOn()
        self._etiquettes()
        self.appliquer_mode(self.mode)

    def appliquer_mode(self, mode):
        """Couleur des tuiles : mosaïque neutre, famille de cuisine ou note."""
        self.mode = mode
        carte = self.carte
        if carte is None or carte.n == 0:
            remplacer(self.a_tuiles, vtk.vtkPolyData())
            return
        fond = JETONS["canvas"]
        if mode == "cuisine":
            couleurs = [melange(fond, famille(r.cuisine)[1], 0.42) for r in carte.restaurants]
        elif mode == "note":
            notes = np.array([r.note if r.note is not None else np.nan for r in carte.restaurants], float)
            bas, haut = (np.nanmin(notes), np.nanmax(notes)) if np.isfinite(notes).any() else (0, 1)
            couleurs = []
            for v in notes:
                if not np.isfinite(v):
                    couleurs.append(rgb(MOSAIQUE[0]))
                    continue
                k = 0 if haut == bas else int(round((v - bas) / (haut - bas) * (len(RAMPE_NOTE) - 1)))
                couleurs.append(melange(fond, RAMPE_NOTE[k], 0.62))
        else:
            couleurs = [rgb(MOSAIQUE[c % len(MOSAIQUE)]) for c in self.coloration]
        self.couleurs_tuiles = couleurs_uint8(couleurs)
        tuiles = vtk.vtkPolyData()
        tuiles.ShallowCopy(carte.voronoi)
        indices = vtk_to_numpy(carte.voronoi.GetCellData().GetArray("indice"))
        tableau = numpy_to_vtk(self.couleurs_tuiles[indices], deep=True)
        tableau.SetName("rgb")
        tuiles.GetCellData().SetScalars(tableau)
        remplacer(self.a_tuiles, tuiles)
        self.a_tuiles.GetMapper().SetScalarModeToUseCellData()
        self.a_tuiles.GetMapper().SetColorModeToDirectScalars()
        self.a_tuiles.GetMapper().ScalarVisibilityOn()
        if self.selection_courante:
            self.selection(*self.selection_courante)

    def calque(self, nom, visible):
        for a in self.calques[nom]:
            a.SetVisibility(visible)

    # -- caméra ------------------------------------------------------------------------

    def unite_par_pixel(self):
        return 2 * self.ren.GetActiveCamera().GetParallelScale() / max(1, self.taille_px[1])

    def cadrer(self, largeur, hauteur):
        """Cadre toute la carte dans la vue (appelé au démarrage et à chaque redimensionnement)."""
        self.taille_px = (max(1, largeur), max(1, hauteur))
        x0, x1, y0, y1 = self.cadre
        cam = self.ren.GetActiveCamera()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cam.SetFocalPoint(cx, cy, 0)
        # La caméra doit rester devant toutes les couches (jusqu'à z = 130), même pour une carte minuscule.
        cam.SetPosition(cx, cy, max(10 * max(x1 - x0, y1 - y0), 1000.0))
        cam.SetViewUp(0, 1, 0)
        aspect = self.taille_px[0] / self.taille_px[1]
        self.echelle_ajustee = 1.04 * max((y1 - y0) / 2, (x1 - x0) / 2 / aspect)
        cam.SetParallelScale(self.echelle_ajustee)
        self.ren.ResetCameraClippingRange()
        self._apres_camera()

    def redimensionner(self, largeur, hauteur):
        if abs(self.ren.GetActiveCamera().GetParallelScale() - self.echelle_ajustee) < 1e-9:
            self.cadrer(largeur, hauteur)
        else:  # l'utilisateur a zoomé : on garde son zoom
            self.taille_px = (max(1, largeur), max(1, hauteur))
            self._apres_camera()

    def ecran_vers_monde(self, sx, sy):
        c = vtk.vtkCoordinate()
        c.SetCoordinateSystemToDisplay()
        c.SetValue(sx, sy, 0)
        x, y, _ = c.GetComputedWorldValue(self.ren)
        return x, y

    def zoomer(self, facteur, sx, sy):
        """Zoom autour du curseur ; jamais plus loin que la carte entière."""
        cam = self.ren.GetActiveCamera()
        x, y = self.ecran_vers_monde(sx, sy)
        echelle = min(self.echelle_ajustee, max(self.echelle_ajustee * 0.04, cam.GetParallelScale() * facteur))
        if echelle >= self.echelle_ajustee:
            self.cadrer(*self.taille_px)
            return
        r = echelle / cam.GetParallelScale()
        fx, fy, _ = cam.GetFocalPoint()
        nx, ny = x + (fx - x) * r, y + (fy - y) * r
        cam.SetParallelScale(echelle)
        self._viser(nx, ny)

    def deplacer(self, dx_px, dy_px):
        cam = self.ren.GetActiveCamera()
        if abs(cam.GetParallelScale() - self.echelle_ajustee) < 1e-9:
            return  # carte entière visible : rien à déplacer
        u = self.unite_par_pixel()
        fx, fy, _ = cam.GetFocalPoint()
        self._viser(fx - dx_px * u, fy - dy_px * u)

    def _viser(self, x, y):
        cam = self.ren.GetActiveCamera()
        x0, x1, y0, y1 = self.cadre
        x, y = min(max(x, x0), x1), min(max(y, y0), y1)
        z = cam.GetPosition()[2]
        cam.SetFocalPoint(x, y, 0)
        cam.SetPosition(x, y, z)
        self.ren.ResetCameraClippingRange()
        self._apres_camera()

    def _apres_camera(self):
        # Les étiquettes et le halo d'impulsion dépendent de la taille d'un pixel.
        if self.carte is not None:
            self._etiquettes()

    LIMITE_NOMS = 150
    CANDIDATS_NOMS = 800

    def _etiquettes(self):
        """Choisit les noms affichés : visibles, sans chevauchement, les mieux notés d'abord.

        Le placeur automatique de VTK refait ce calcul à chaque image (250 ms avec 10 000 noms) ;
        ici il n'a lieu qu'au zoom, au déplacement ou au redimensionnement. En zoomant, d'autres
        noms trouvent de la place et apparaissent.
        """
        carte = self.carte
        if carte is None or carte.n == 0:
            self.pts_noms.ShallowCopy(vtk.vtkPolyData())
            return
        cam = self.ren.GetActiveCamera()
        fx, fy, _ = cam.GetFocalPoint()
        demi_h = cam.GetParallelScale()
        demi_l = demi_h * self.taille_px[0] / self.taille_px[1]
        visibles = np.flatnonzero((np.abs(carte.sites[:, 0] - fx) <= demi_l)
                                  & (np.abs(carte.sites[:, 1] - fy) <= demi_h))
        notes = np.array([r.note if r.note is not None else 0.0 for r in carte.restaurants])
        candidats = visibles[np.argsort(-notes[visibles], kind="stable")[:self.CANDIDATS_NOMS]]
        u = self.unite_par_pixel()
        decalage = 15 * u  # nom posé au-dessus du marqueur
        # Carte dense vue de loin : peu de noms, sinon ils noient les cellules ; le zoom en révèle d'autres.
        limite = self.LIMITE_NOMS if len(visibles) <= 400 else 40
        boites = np.empty((0, 4))
        retenus = []
        for k in candidats:
            # Largeur estimée de Segoe UI 12 px (6,8 px par caractère) plus une petite marge.
            demi_largeur = (6.8 * len(carte.restaurants[k].nom) + 10) * u / 2
            x, y = carte.sites[k][0], carte.sites[k][1] + decalage
            boite = (x - demi_largeur, x + demi_largeur, y - 9 * u, y + 9 * u)
            chevauche = ((boites[:, 0] < boite[1]) & (boites[:, 1] > boite[0])
                         & (boites[:, 2] < boite[3]) & (boites[:, 3] > boite[2]))
            if not chevauche.any():
                boites = np.vstack([boites, boite])
                retenus.append(k)
                if len(retenus) >= limite:
                    break
        pts = poly_points(carte.sites[retenus] + [0, decalage] if retenus else np.empty((0, 2)), z=130.0)
        noms = vtk.vtkStringArray()
        noms.SetName("nom")
        for k in retenus:
            noms.InsertNextValue(carte.restaurants[k].nom)
        pts.GetPointData().AddArray(noms)
        self.pts_noms.ShallowCopy(pts)

    # -- sélection ---------------------------------------------------------------------

    def selection(self, x, y, i, voisins):
        """Prépare la sélection (position, plus proche i, voisins) ; animer_selection(t) la dévoile."""
        self.selection_courante = (x, y, i, list(voisins))
        carte = self.carte
        tuile = carte.tuile(i)
        couleur = self.couleurs_tuiles[i] / 255
        remplacer(self.a_selection, poly_polygones([tuile]))
        self.a_selection.GetProperty().SetColor(melange(couleur, JETONS["ink"], 0.16))
        remplacer(self.a_contour, poly_lignes([tuile], fermees=True))
        remplacer(self.a_halo, poly_lignes([tuile], fermees=True))
        remplacer(self.a_voisines, poly_polygones([carte.tuile(j) for j in voisins]))
        remplacer(self.a_marq_voisins, poly_points(carte.sites[list(voisins)]) if voisins else vtk.vtkPolyData())
        sommets = [(cx, cy) for cx, cy, _ in self._fleur(i)]
        remplacer(self.a_sommets, poly_points(sommets) if sommets else vtk.vtkPolyData())
        remplacer(self.a_user, poly_points([(x, y)]))
        remplacer(self.a_user_anneau, poly_points([(x, y)]))
        sx, sy = carte.sites[i]
        u = self.unite_par_pixel()
        remplacer(self.a_lien_user, poly_segments(pointilles((x, y), (sx, sy), 7 * u, 5 * u)))
        d = carte.distance(i, x, y)
        self.a_distance.SetInput(f" {d:.0f} ")
        self.a_distance.GetPositionCoordinate().SetValue((x + sx) / 2, (y + sy) / 2, 120.0)
        self.a_distance.SetVisibility(d > 40 * u)  # inutile si l'utilisateur est sur le site
        for a in (self.a_selection, self.a_contour, self.a_halo, self.a_voisines, self.a_marq_voisins,
                  self.a_user, self.a_user_anneau, self.a_lien_user):
            a.VisibilityOn()
        self.animer_selection(1.0)

    def animer_selection(self, t):
        """t = 0 : rien ; t = 1 : liens vers les voisins tracés et fleur de cercles vides ouverte."""
        if not self.selection_courante:
            return
        x, y, i, voisins = self.selection_courante
        carte = self.carte
        sx, sy = carte.sites[i]
        liens = [((sx, sy), (sx + (carte.sites[j][0] - sx) * t, sy + (carte.sites[j][1] - sy) * t))
                 for j in voisins]
        remplacer(self.a_liens, poly_segments(liens))
        cercles = [cercle(cx, cy, r * t) for cx, cy, r in self._fleur(i)]
        remplacer(self.a_fleur, poly_lignes(cercles, fermees=True) if cercles else vtk.vtkPolyData())
        self.a_halo.GetProperty().SetOpacity(0.25 * t)

    def _fleur(self, i):
        """Cercles vides autour de i dont le centre est un sommet visible de sa cellule.

        Sur le bord, un triangle très plat a un cercle immense centré hors du cadre :
        il est juste mais illisible, et son centre n'est pas un sommet de la cellule affichée.
        """
        x0, x1, y0, y1 = self.cadre
        return [(cx, cy, r) for cx, cy, r in self.carte.cercles(i) if x0 <= cx <= x1 and y0 <= cy <= y1]

    def pulse(self, t):
        """Onde qui part de la position de l'utilisateur (retour visuel du clic)."""
        if not self.selection_courante or t >= 1:
            self.a_pulse.VisibilityOff()
            return
        x, y = self.selection_courante[:2]
        rayon = (10 + 46 * t) * self.unite_par_pixel()
        remplacer(self.a_pulse, poly_lignes([cercle(x, y, rayon, 96)], fermees=True))
        self.a_pulse.GetProperty().SetOpacity(0.9 * (1 - t) ** 1.5)
        self.a_pulse.VisibilityOn()

    def effacer_selection(self):
        self.selection_courante = None
        for a in (self.a_selection, self.a_contour, self.a_halo, self.a_voisines, self.a_marq_voisins,
                  self.a_user, self.a_user_anneau, self.a_lien_user, self.a_distance, self.a_pulse):
            a.VisibilityOff()
        for a in (self.a_liens, self.a_fleur, self.a_sommets):
            remplacer(a, vtk.vtkPolyData())
        self.dual(None, None)

    def survol(self, i):
        if i is None or self.carte is None:
            self.a_survol.VisibilityOff()
            self.a_survol_bord.VisibilityOff()
            return
        tuile = self.carte.tuile(i)
        remplacer(self.a_survol, poly_polygones([tuile]))
        remplacer(self.a_survol_bord, poly_lignes([tuile], fermees=True))
        self.a_survol.VisibilityOn()
        self.a_survol_bord.VisibilityOn()

    def dual(self, i, j):
        """Montre la dualité : frontière de Voronoï commune à i et j, et l'arête de Delaunay entre i et j."""
        frontiere = self.carte.frontiere(i, j) if (self.carte is not None and i is not None and j is not None) else None
        if frontiere is None:
            self.a_dual_vor.VisibilityOff()
            self.a_dual_del.VisibilityOff()
            return
        u = self.unite_par_pixel()
        remplacer(self.a_dual_vor, poly_segments([frontiere]))
        remplacer(self.a_dual_del, poly_segments(pointilles(self.carte.sites[i], self.carte.sites[j], 6 * u, 4 * u)))
        self.a_dual_vor.VisibilityOn()
        self.a_dual_del.VisibilityOn()

    # -- trajet (extension A) ----------------------------------------------------------

    def trajet_effacer(self):
        self.trajet_points, self.trajet_croisements = [], []
        remplacer(self.a_trajet, vtk.vtkPolyData())
        remplacer(self.a_croisements, vtk.vtkPolyData())

    def trajet_ajouter(self, x, y, croisements=()):
        self.trajet_points.append((x, y))
        self.trajet_croisements.extend((cx, cy) for cx, cy, *_ in croisements)
        if len(self.trajet_points) > 1:
            remplacer(self.a_trajet, poly_lignes([self.trajet_points]))
        if self.trajet_croisements:
            remplacer(self.a_croisements, poly_points(self.trajet_croisements))


# --- Vues 3D ---------------------------------------------------------------------------

def _renderer_3d():
    ren = vtk.vtkRenderer()
    ren.SetBackground(rgb(JETONS["canvas"]))
    ren.GetActiveCamera().ParallelProjectionOn()
    kit = vtk.vtkLightKit()
    kit.SetKeyLightIntensity(0.9)
    kit.AddLightsToRenderer(ren)
    return ren


def _materiau(a, speculaire=0.25):
    p = a.GetProperty()
    p.LightingOn()
    p.SetAmbient(0.18)
    p.SetDiffuse(0.85)
    p.SetSpecular(speculaire)
    p.SetSpecularPower(24)


def _colorer_cellules(poly, nom_tableau, couleurs_sites):
    """Copie de `poly` dont chaque cellule prend la couleur de son site."""
    sites = vtk_to_numpy(poly.GetCellData().GetArray(nom_tableau)).astype(int)
    copie = vtk.vtkPolyData()
    copie.ShallowCopy(poly)
    tableau = numpy_to_vtk(couleurs_sites[sites], deep=True)
    tableau.SetName("rgb")
    copie.GetCellData().SetScalars(tableau)
    return copie


def enveloppe_cellules(carte, cadre, pente, pas):
    """Chaque cône restreint à sa propre cellule de Voronoï.

    Au-dessus de sa cellule, le cône d'un restaurant est le plus haut : l'union de ces morceaux
    est donc exactement la surface visible (l'enveloppe), sans les nappes cachées qui
    encombreraient les côtés du bloc. Un cône est réglé (droites issues du sommet) : un éventail
    de triangles depuis le sommet vers le bord subdivisé de la cellule le décrit fidèlement.
    Renvoie (surface, murs) avec le tableau de cellules "site" ; les murs ferment le bloc au cadre.
    """
    x0, x1, y0, y1 = cadre
    points, triangles, sites_tri = [], [], []
    bords_cadre = []
    vus = set()
    for i in range(carte.n):
        sx, sy = carte.sites[i]
        if (sx, sy) in vus:  # doublons exacts : une seule fois la même cellule
            continue
        vus.add((sx, sy))
        tuile = carte.tuile(i)
        bord = []
        for k in range(len(tuile)):
            p, q = tuile[k], tuile[(k + 1) % len(tuile)]
            m = max(2, int(math.ceil(math.hypot(*(q - p)) / pas)))
            bord.extend(p + (q - p) * s / m for s in range(m))
            for axe, limite in ((0, x0), (0, x1), (1, y0), (1, y1)):
                if abs(p[axe] - limite) < 1e-9 and abs(q[axe] - limite) < 1e-9:
                    bords_cadre.append((p, q, i))
        sommet = len(points)
        points.append((sx, sy, 0.0))
        for bx, by in bord:
            points.append((bx, by, -pente * math.hypot(bx - sx, by - sy)))
        m = len(bord)
        for k in range(m):
            triangles.append((sommet, sommet + 1 + k, sommet + 1 + (k + 1) % m))
            sites_tri.append(i)

    surface = _polydata_triangles(points, triangles, sites_tri)
    zs = np.array([p[2] for p in points]) if points else np.zeros(1)
    base = zs.min() - 0.06 * max(x1 - x0, y1 - y0)
    pts_murs, tri_murs, sites_murs = [], [], []
    for p, q, i in bords_cadre:
        sx, sy = carte.sites[i]
        m = max(2, int(math.ceil(math.hypot(*(q - p)) / pas)))
        for s in range(m):
            a, b = p + (q - p) * s / m, p + (q - p) * (s + 1) / m
            za, zb = (-pente * math.hypot(c[0] - sx, c[1] - sy) for c in (a, b))
            k = len(pts_murs)
            pts_murs.extend([(a[0], a[1], za), (b[0], b[1], zb), (b[0], b[1], base), (a[0], a[1], base)])
            tri_murs.extend([(k, k + 1, k + 2), (k, k + 2, k + 3)])
            sites_murs.extend([i, i])
    return surface, _polydata_triangles(pts_murs, tri_murs, sites_murs), base


def _polydata_triangles(points, triangles, sites):
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    if points:
        pts.SetData(numpy_to_vtk(np.asarray(points, dtype=float), deep=True))
    poly.SetPoints(pts)
    cellules = vtk.vtkCellArray()
    for t in triangles:
        cellules.InsertNextCell(3, t)
    poly.SetPolys(cellules)
    tableau = numpy_to_vtk(np.asarray(sites, dtype=np.int32), deep=True)
    tableau.SetName("site")
    poly.GetCellData().AddArray(tableau)
    normales = vtk.vtkPolyDataNormals()
    normales.SetInputData(poly)
    normales.SetFeatureAngle(25)  # arêtes vives aux crêtes, cônes lisses
    normales.SplittingOn()
    normales.ConsistencyOn()
    normales.Update()
    return normales.GetOutput()


class Orbite:
    """Caméra orthographique qui passe de la vue de dessus (t = 0) à une vue inclinée (t = 1)."""

    def __init__(self, ren, elevation=36.0, azimut=-30.0):
        self.ren, self.elevation, self.azimut = ren, elevation, azimut
        self.centre, self.rayon, self.taille_px = (0, 0, 0), 1.0, (800, 600)
        self.cadre = (0, 1, 0, 1)
        self.t = 0.0

    def regler(self, cadre, profondeur, taille_px):
        x0, x1, y0, y1 = cadre
        self.cadre = cadre
        # profondeur > 0 : la scène descend sous z = 0 (cônes) ; < 0 : elle monte (paraboloïde).
        self.centre = ((x0 + x1) / 2, (y0 + y1) / 2, -profondeur / 2)
        self.rayon = 10 * max(x1 - x0, y1 - y0, abs(profondeur))
        self.profondeur = profondeur
        self.taille_px = taille_px
        self.poser(self.t)

    def poser(self, t):
        self.t = t
        e = math.radians(90 - (90 - self.elevation) * t)
        a = math.radians(self.azimut * t)
        cx, cy, cz = self.centre
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx + self.rayon * math.cos(e) * math.sin(a),
                        cy - self.rayon * math.cos(e) * math.cos(a),
                        cz + self.rayon * math.sin(e))
        cam.SetViewUp(-math.sin(e) * math.sin(a), math.sin(e) * math.cos(a), math.cos(e))
        x0, x1, y0, y1 = self.cadre
        aspect = self.taille_px[0] / max(1, self.taille_px[1])
        dessus = max((y1 - y0) / 2, (x1 - x0) / 2 / aspect)
        # En vue inclinée la profondeur du bloc apparaît : on recule un peu pour tout garder.
        cam.SetParallelScale(1.04 * dessus * (1 + 0.32 * t))
        self.ren.ResetCameraClippingRange()


class VueCones:
    """Un cône de distance par restaurant (Hoff et al., 1999) : vu de dessus, c'est le Voronoï."""

    PENTE = 0.45

    def __init__(self):
        self.ren = _renderer_3d()
        self.orbite = Orbite(self.ren)
        self.a_cones = acteur(eclairage=True)
        _materiau(self.a_cones)
        self.a_murs = acteur(eclairage=True)
        _materiau(self.a_murs, 0.05)
        # Les crêtes coïncident avec les arêtes de la surface : un léger décalage vertical
        # (réglé dans charger) leur évite d'être masquées par le test de profondeur.
        self.a_cretes = acteur(couleur="ink", opacite=0.85, largeur=1.5)
        self.a_sommets = acteur(couleur="ink")
        self.a_sommets.GetProperty().SetPointSize(7)
        self.a_sommets.GetProperty().RenderPointsAsSpheresOn()
        self.a_choisi = acteur(couleur="accent2", largeur=3.0)
        self.a_user = acteur(couleur="accent2", largeur=2.0)
        self.acteurs = (self.a_cones, self.a_murs, self.a_cretes, self.a_sommets, self.a_choisi, self.a_user)
        for a in self.acteurs:
            self.ren.AddViewProp(a)
        self.carte = None
        self.echelle = 1.0

    def charger(self, carte, cadre, taille_px):
        self.carte = carte
        for a in self.acteurs:
            remplacer(a, vtk.vtkPolyData())
        if carte.n == 0:
            self.orbite.regler(cadre, 1.0, taille_px)
            return
        couleurs = couleurs_uint8([famille(r.cuisine)[1] for r in carte.restaurants])
        sombres = (couleurs * 0.45).astype(np.uint8)  # murs plus sombres : le dessus reste la carte
        x0, x1, y0, y1 = cadre
        surface, murs, base = enveloppe_cellules(carte, cadre, self.PENTE, math.hypot(x1 - x0, y1 - y0) / 260)
        remplacer(self.a_cones, _colorer_cellules(surface, "site", couleurs))
        remplacer(self.a_murs, _colorer_cellules(murs, "site", sombres))
        for a in (self.a_cones, self.a_murs):
            a.GetMapper().SetScalarModeToUseCellData()
            a.GetMapper().SetColorModeToDirectScalars()
            a.GetMapper().ScalarVisibilityOn()
        if carte.n >= 2:
            remplacer(self.a_cretes, scenes3d.cretes(carte.segments_voronoi(), carte.sites, pente=self.PENTE))
            self.a_cretes.GetMapper().ScalarVisibilityOff()
        # Décalages verticaux proportionnels à la carte : justes à toute échelle de coordonnées.
        self.echelle = max(x1 - x0, y1 - y0)
        self.a_cretes.SetPosition(0, 0, 0.0015 * self.echelle)
        remplacer(self.a_sommets, poly_points(carte.sites, z=0.0005 * self.echelle))
        self.orbite.regler(cadre, -base, taille_px)
        self.montrer(None)

    def montrer(self, selection):
        """Met en valeur la cellule du plus proche sur l'enveloppe et la verticale de l'utilisateur."""
        if not selection or self.carte is None:
            self.a_choisi.VisibilityOff()
            self.a_user.VisibilityOff()
            return
        x, y, i, _ = selection
        sx, sy = self.carte.sites[i]
        tuile = self.carte.tuile(i)
        # Le bord de la tuile posé sur le cône de i : z = -pente * distance au site.
        bord = []
        for k in range(len(tuile)):
            p, q = tuile[k], tuile[(k + 1) % len(tuile)]
            for s in np.linspace(0, 1, 16, endpoint=False):
                px, py = p[0] + (q[0] - p[0]) * s, p[1] + (q[1] - p[1]) * s
                bord.append((px, py, -self.PENTE * math.hypot(px - sx, py - sy) + 0.0005 * self.echelle))
        remplacer(self.a_choisi, poly_lignes([bord], fermees=True))
        dz = -self.PENTE * math.hypot(x - sx, y - sy)
        remplacer(self.a_user, poly_lignes([[(x, y, 0.04 * self.echelle), (x, y, dz)]]))
        self.a_choisi.VisibilityOn()
        self.a_user.VisibilityOn()


class VueParaboloide:
    """Relèvement sur z = u² + v² : Delaunay = enveloppe convexe inférieure des points relevés."""

    PROPORTION = 0.32  # hauteur du paraboloïde rapportée à la taille de la carte (indépendant de l'échelle)

    def __init__(self):
        self.ren = _renderer_3d()
        self.orbite = Orbite(self.ren, elevation=24.0, azimut=-34.0)
        self.a_surface = acteur(couleur="ink3", opacite=0.10, eclairage=True)
        _materiau(self.a_surface, 0.1)
        self.a_faces = acteur(couleur="accent", opacite=0.38, eclairage=True)
        _materiau(self.a_faces, 0.3)
        self.a_aretes = acteur(couleur="accent2", largeur=1.4)
        self.a_tiges = acteur(couleur="ink4", opacite=0.7, largeur=1.0)
        self.a_points = acteur(couleur="ink")
        self.a_points.GetProperty().SetPointSize(7)
        self.a_points.GetProperty().RenderPointsAsSpheresOn()
        self.a_plan = acteur(couleur="ink3", opacite=0.55, largeur=1.0)
        self.a_plan.GetProperty().SetRepresentationToWireframe()  # la triangulation à plat, en filaire
        self.a_cercle = acteur(couleur="accent2", largeur=2.2)
        self.a_cercle_releve = acteur(couleur="ink", largeur=2.2)
        self.a_face_choisie = acteur(couleur="accent2", opacite=0.75, eclairage=True)
        self.releves = (self.a_surface, self.a_faces, self.a_aretes, self.a_tiges, self.a_points,
                        self.a_cercle_releve, self.a_face_choisie)
        for a in (self.a_plan, self.a_cercle) + self.releves:
            self.ren.AddViewProp(a)
        self.carte = None

    def charger(self, carte, cadre, taille_px):
        self.carte, self.cadre = carte, cadre
        x0, x1, y0, y1 = cadre
        self.hauteur = self.PROPORTION * max(x1 - x0, y1 - y0)
        self.a_cercle.SetPosition(0, 0, 0.0005 * max(x1 - x0, y1 - y0))  # au-dessus du plan filaire
        if carte.n < 3 or len(carte.triangles) == 0:
            for a in (self.a_plan, self.a_cercle) + self.releves:
                remplacer(a, vtk.vtkPolyData())
            self.orbite.regler(cadre, -self.hauteur, taille_px)
            return
        geo = scenes3d.paraboloide(carte.sites, carte.triangles, cadre, hauteur=self.hauteur)
        for cle, a in (("surface", self.a_surface), ("faces", self.a_faces), ("aretes", self.a_aretes),
                       ("tiges", self.a_tiges), ("points", self.a_points), ("plan", self.a_plan)):
            remplacer(a, geo[cle])
            a.GetMapper().ScalarVisibilityOff()
        self.orbite.regler(cadre, -self.hauteur, taille_px)
        self.exemple(None)

    def exemple(self, position):
        """Triangle de démonstration : celui qui contient la position (ou le premier).

        Son cercle circonscrit, relevé sur le paraboloïde, reste dans le plan de la face :
        un point est dans le cercle si et seulement si son relevé passe sous ce plan.
        """
        carte = self.carte
        if carte is None or len(carte.triangles) == 0:
            return False
        k = _triangle_contenant(carte.sites, carte.triangles, position) if position is not None else None
        contient = k is not None
        k = k if contient else 0
        tri = carte.triangles[k]
        cx, cy, r = carte.cercles_circonscrits()[k]
        anneau = cercle(cx, cy, r, 160)
        remplacer(self.a_cercle, poly_lignes([anneau], fermees=True))
        a, b, c, d = scenes3d.plan_triangle(carte.sites, tri, self.cadre, hauteur=self.hauteur)
        releve = [(px, py, -(a * px + b * py + d) / c) for px, py in anneau]
        remplacer(self.a_cercle_releve, poly_lignes([releve], fermees=True))
        sommets = [(carte.sites[v][0], carte.sites[v][1], -(a * carte.sites[v][0] + b * carte.sites[v][1] + d) / c)
                   for v in tri]
        face = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        for s in sommets:
            pts.InsertNextPoint(*s)
        cellule = vtk.vtkCellArray()
        cellule.InsertNextCell(3, (0, 1, 2))
        face.SetPoints(pts)
        face.SetPolys(cellule)
        remplacer(self.a_face_choisie, face)
        return contient

    def lever(self, t):
        """Anime le relèvement : t = 0 tout est à plat, t = 1 points sur le paraboloïde."""
        for a in self.releves:
            a.SetScale(1, 1, max(t, 1e-3))


def _triangle_contenant(sites, triangles, position):
    x, y = position
    for k, (a, b, c) in enumerate(triangles):
        (x1, y1), (x2, y2), (x3, y3) = sites[a], sites[b], sites[c]
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if det == 0:
            continue
        l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
        l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
        if l1 >= 0 and l2 >= 0 and l1 + l2 <= 1:
            return k
    return None
