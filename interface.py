"""Interface PySide6 : fenêtre, panneaux, interactions souris et animations.

Organisation :
    barre du haut      choix de la vue (carte 2D, cônes 3D, paraboloïde 3D), capture, aide
    panneau de gauche  filtres (cuisines, note, prix), calques, couleurs, jeu de données
    centre             vue VTK
    panneau de droite  résultat de la recherche, ou explication de la vue 3D
"""
import math
import time
from pathlib import Path

import vtkmodules.qt

vtkmodules.qt.PyQtImpl = "PySide6"
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402

import numpy as np  # noqa: E402
import vtk  # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy  # noqa: E402
from PySide6.QtCore import (QElapsedTimer, QEvent, QPoint, QPointF, QPropertyAnimation, QRectF, QSize, Qt,  # noqa: E402
                            QTimer, Signal)
from PySide6.QtGui import (QColor, QCursor, QFontMetrics, QImage, QKeySequence, QPainter, QPen,  # noqa: E402
                           QPolygonF, QShortcut)
from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QApplication, QButtonGroup, QDialog, QFileDialog,  # noqa: E402
                               QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout, QMainWindow, QMessageBox,
                               QPushButton, QScrollArea, QSizePolicy, QSlider, QStackedWidget, QVBoxLayout,
                               QWidget)

import geometrie  # noqa: E402
from vues import FAMILLES, JETONS, Carte2D, VueCones, VueParaboloide, famille  # noqa: E402

THEME = Path(__file__).with_name("theme.qss")
CONSEIL_CSV = ("Vérifiez le fichier : colonnes ID, nom, X et Y obligatoires (type, prix, note et avis "
               "facultatifs), séparateur virgule ou point-virgule, encodage UTF-8. Corrigez la ligne indiquée "
               "puis rouvrez le fichier.")


def qc(nom, alpha=255):
    c = QColor(JETONS.get(nom, nom))
    c.setAlpha(alpha)
    return c


PRECISION = [0]  # décimales des distances et positions, selon la taille de la carte (voir reconstruire)


def nombre(v, decimales=None):
    """Format français : virgule décimale, espace fine pour les milliers."""
    decimales = PRECISION[0] if decimales is None else decimales
    return f"{v:,.{decimales}f}".replace(",", " ").replace(".", ",")


def pluriel(n, mot):
    return f"{nombre(n, 0)} {mot}{'s' if n > 1 else ''}"


def typo(texte):
    """Typographie française des textes de l'interface : espace insécable avant « : », apostrophe courbe."""
    return texte.replace(" :", " :").replace("'", "’")


def etiquette(texte="", role=None):
    # Les noms viennent des données : on ne les retouche pas.
    lbl = QLabel(texte if role in ("nom", "nom-principal", "touche", "valeur") else typo(texte))
    if role:
        lbl.setProperty("role", role)
    return lbl


def html_prix(prix):
    if not prix:
        return f"<span style='color:{JETONS['ink3']}'>prix ?</span>"
    return (f"<span style='color:{JETONS['ink']}'>{'€' * prix}</span>"
            f"<span style='color:{JETONS['ink4']}'>{'€' * (4 - prix)}</span>")


def html_note(note):
    if note is None:
        return f"<span style='color:{JETONS['ink3']}'>sans note</span>"
    return f"<span style='color:{JETONS['ink']}'>{nombre(note, 1)}</span> <span style='color:{JETONS['ink3']}'>★</span>"


def aire(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# --- Animations --------------------------------------------------------------------------

def sortie(t):
    return 1 - (1 - t) ** 3


def douce(t):
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def animations_reduites():
    """Vrai si « Afficher les animations dans Windows » est désactivé (réglage d'accessibilité)."""
    try:
        import ctypes
        actif = ctypes.c_int(1)
        ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(actif), 0)  # SPI_GETCLIENTAREAANIMATION
        return not actif.value
    except (AttributeError, OSError):
        return False


class Animateur:
    """Petites animations à 60 images/s : chaque animation reçoit t dans [0, 1].

    Si l'utilisateur a désactivé les animations dans Windows, chaque animation saute à son état final.
    """

    def __init__(self, rendre):
        self.rendre = rendre
        self.anims = {}
        self.reduites = animations_reduites()
        self.horloge = QElapsedTimer()
        self.horloge.start()
        self.minuterie = QTimer()
        self.minuterie.setInterval(16)
        self.minuterie.timeout.connect(self._tic)

    def lancer(self, cle, duree, maj, courbe=sortie, fin=None):
        if self.reduites:
            maj(1.0)
            if fin:
                fin()
            self.rendre()
            return
        self.anims[cle] = (self.horloge.elapsed(), duree * 1000, maj, courbe, fin)
        maj(courbe(0.0))
        if not self.minuterie.isActive():
            self.minuterie.start()

    def arreter(self, cle):
        self.anims.pop(cle, None)

    def _tic(self):
        maintenant = self.horloge.elapsed()
        for cle, (debut, duree, maj, courbe, fin) in list(self.anims.items()):
            t = min(1.0, (maintenant - debut) / duree)
            maj(courbe(t))
            if t >= 1.0:
                self.anims.pop(cle, None)
                if fin:
                    fin()
        self.rendre()
        if not self.anims:
            self.minuterie.stop()


# --- Petits composants -------------------------------------------------------------------

class Marque(QWidget):
    """Marque de l'application : une cellule de Voronoï et son site."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(22, 22)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cellule = QPolygonF([QPointF(11, 1.5), QPointF(20, 6.5), QPointF(18.5, 16.5), QPointF(9.5, 20.5),
                             QPointF(2.5, 14), QPointF(3.5, 5)])
        p.setPen(QPen(qc("accent2"), 1.6))
        p.setBrush(qc("accent", 70))
        p.drawPolygon(cellule)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc("ink"))
        p.drawEllipse(QPointF(11, 11.5), 2.5, 2.5)


class Segments(QFrame):
    """Contrôle segmenté (onglets en pilule)."""

    change = Signal(str)

    def __init__(self, options, actif=None):
        super().__init__()
        self.setProperty("role", "segments")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.groupe = QButtonGroup(self)
        self.boutons = {}
        for cle, libelle in options:
            b = QPushButton(libelle)
            b.setProperty("role", "segment")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            b.clicked.connect(lambda _=False, c=cle: self.change.emit(c))
            self.groupe.addButton(b)
            lay.addWidget(b)
            self.boutons[cle] = b
        self.boutons[actif or options[0][0]].setChecked(True)

    def choisir(self, cle):
        self.boutons[cle].setChecked(True)


class Interrupteur(QAbstractButton):
    def __init__(self, coche=True):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(coche)
        self.setFixedSize(34, 20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        piste = QRectF(1, 1, 32, 18)
        if self.isChecked():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc("accent"))
        else:
            p.setPen(QPen(qc("hairline3"), 1))
            p.setBrush(qc("s4"))
        p.drawRoundedRect(piste, 9, 9)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc("ink") if self.isChecked() else qc("ink3"))
        p.drawEllipse(QPointF(24 if self.isChecked() else 10, 10), 6.5, 6.5)
        if self.hasFocus():
            p.setPen(QPen(qc("accent2"), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(piste.adjusted(-1, -1, 1, 1), 10, 10)


class Puce(QAbstractButton):
    """Filtre de cuisine : la pastille donne la couleur de famille (légende et filtre en un)."""

    def __init__(self, cuisine, couleur, compte):
        super().__init__()
        self.cuisine, self.couleur, self.compte = cuisine, couleur, compte
        self.setCheckable(True)
        self.setChecked(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setMinimumHeight(30)
        self.setToolTip(typo(f"{cuisine} : famille « {famille(cuisine)[0]} »"))
        self.setAccessibleName(f"{cuisine}, {compte} restaurants")

    def sizeHint(self):
        self.ensurePolished()  # police du thème appliquée avant de mesurer le texte
        fm = QFontMetrics(self.font())
        return QSize(26 + fm.horizontalAdvance(self.cuisine) + 12 + fm.horizontalAdvance(str(self.compte)) + 12, 30)

    def enterEvent(self, e):
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.update()
        super().leaveEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        actif = self.isChecked()
        bord = qc("hairline3") if self.underMouse() else qc("hairline2" if actif else "hairline")
        if self.hasFocus():
            bord = qc("accent2")
        p.setPen(QPen(bord, 1))
        p.setBrush(qc("s3" if actif else "s1"))
        p.drawRoundedRect(r, 6, 6)
        pastille = QColor(self.couleur)
        pastille.setAlpha(255 if actif else 70)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(pastille)
        h = self.height()
        p.drawRoundedRect(QRectF(10, h / 2 - 4, 8, 8), 2, 2)
        p.setPen(qc("ink" if actif else "ink4"))
        place_compte = QFontMetrics(self.font()).horizontalAdvance(str(self.compte)) + 20
        p.drawText(QRectF(26, 0, self.width() - 26 - place_compte, h), Qt.AlignmentFlag.AlignVCenter, self.cuisine)
        p.setPen(qc("ink3" if actif else "ink4"))
        p.drawText(QRectF(0, 0, self.width() - 10, h),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, str(self.compte))


class Flux(QLayout):
    """Disposition en lignes qui se replient (Qt n'en fournit pas) : les puces gardent leur largeur."""

    def __init__(self, espace=6):
        super().__init__()
        self.elements, self.espace = [], espace

    def addItem(self, element):
        self.elements.append(element)

    def count(self):
        return len(self.elements)

    def itemAt(self, k):
        return self.elements[k] if 0 <= k < len(self.elements) else None

    def takeAt(self, k):
        return self.elements.pop(k) if 0 <= k < len(self.elements) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, largeur):
        return self._placer(QRect(0, 0, largeur, 0), simuler=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._placer(rect, simuler=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        taille = QSize()
        for e in self.elements:
            taille = taille.expandedTo(e.minimumSize())
        return taille

    def _placer(self, rect, simuler):
        x, y, hauteur = rect.x(), rect.y(), 0
        for e in self.elements:
            t = e.sizeHint()
            if x + t.width() > rect.right() + 1 and hauteur:
                x, y, hauteur = rect.x(), y + hauteur + self.espace, 0
            if not simuler:
                e.setGeometry(QRect(QPoint(x, y), t))
            x += t.width() + self.espace
            hauteur = max(hauteur, t.height())
        return y + hauteur - rect.y()


class Pastille(QWidget):
    def __init__(self, couleur, taille=10):
        super().__init__()
        self.couleur = couleur
        self.setFixedSize(taille, taille)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.couleur))
        p.drawRoundedRect(QRectF(self.rect()), 2.5, 2.5)


def ligne_option(libelle, touche, coche, rappel):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 3, 0, 3)
    lay.addWidget(etiquette(libelle, "texte"))
    lay.addStretch(1)
    if touche:
        lay.addWidget(etiquette(touche, "touche"))
        lay.addSpacing(8)
    inter = Interrupteur(coche)
    inter.setAccessibleName(libelle)
    inter.toggled.connect(rappel)
    lay.addWidget(inter)
    w.interrupteur = inter
    return w


def section(titre):
    lbl = etiquette(titre, "section")
    lbl.setContentsMargins(0, 18, 0, 6)
    return lbl


class CarteResto(QFrame):
    """Carte d'un restaurant : le plus proche (grande) ou une alternative (compacte)."""

    survole = Signal(int, bool)
    clique = Signal(int)

    def __init__(self, indice, resto, distance, principal=False, ecart=None, details=None):
        super().__init__()
        self.indice = indice
        self.setProperty("role", "carte-principale" if principal else "carte")
        self.setAccessibleName(f"{resto.nom}, {resto.cuisine}, distance {nombre(distance)}")
        if not principal:  # une alternative est une action : souris, Entrée ou Espace
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setToolTip(typo("Aller à ce restaurant et voir ses propres voisins (Entrée)"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14) if principal else lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4 if not principal else 6)
        haut = QHBoxLayout()
        haut.setSpacing(10)
        haut.addWidget(Pastille(famille(resto.cuisine)[1], 12 if principal else 10))
        nom = etiquette(resto.nom, "nom-principal" if principal else "nom")
        nom.setWordWrap(True)
        haut.addWidget(nom, 1)
        dist = etiquette(nombre(distance), "distance-principale" if principal else "distance")
        dist.setToolTip("Distance à votre position")
        haut.addWidget(dist)
        lay.addLayout(haut)
        meta = QLabel(f"<span style='color:{JETONS['ink2']}'>{resto.cuisine}</span>"
                      f"&nbsp;&nbsp;&nbsp;{html_prix(resto.prix)}&nbsp;&nbsp;&nbsp;{html_note(resto.note)}"
                      + (f"&nbsp;&nbsp;<span style='color:{JETONS['ink3']}'>({resto.avis} avis)</span>"
                         if resto.avis else ""))
        meta.setTextFormat(Qt.TextFormat.RichText)
        meta.setContentsMargins(22 if principal else 20, 0, 0, 0)
        if ecart is not None:
            ligne = QHBoxLayout()
            ligne.addWidget(meta, 1)
            plus = etiquette(f"+{nombre(ecart)}", "discret")
            plus.setToolTip("Distance en plus par rapport au plus proche")
            ligne.addWidget(plus)
            lay.addLayout(ligne)
        else:
            lay.addWidget(meta)
        if details:
            grille = QGridLayout()
            grille.setContentsMargins(22, 8, 0, 0)
            grille.setHorizontalSpacing(12)
            grille.setVerticalSpacing(4)
            for k, (cle, valeur) in enumerate(details):
                grille.addWidget(etiquette(cle, "discret"), k, 0)
                grille.addWidget(etiquette(valeur, "texte"), k, 1)
            grille.setColumnStretch(1, 1)
            lay.addLayout(grille)

    def enterEvent(self, e):
        self.survole.emit(self.indice, True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.survole.emit(self.indice, False)
        super().leaveEvent(e)

    def focusInEvent(self, e):
        self.survole.emit(self.indice, True)
        super().focusInEvent(e)

    def focusOutEvent(self, e):
        self.survole.emit(self.indice, False)
        super().focusOutEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.focusPolicy() != Qt.FocusPolicy.NoFocus:
            self.clique.emit(self.indice)
        super().mousePressEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clique.emit(self.indice)
        else:
            super().keyPressEvent(e)


def zone_defilante(contenu):
    zone = QScrollArea()
    zone.setWidgetResizable(True)
    zone.setFrameShape(QFrame.Shape.NoFrame)
    zone.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    zone.setWidget(contenu)
    return zone


# --- Panneau de gauche : filtres et affichage --------------------------------------------

class PanneauExplorer(QFrame):
    filtres = Signal()
    calque = Signal(str, bool)
    mode = Signal(str)
    donnees = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("explorer")
        self.setFixedWidth(300)
        externe = QVBoxLayout(self)
        externe.setContentsMargins(0, 0, 0, 0)
        contenu = QWidget()
        self.lay = QVBoxLayout(contenu)
        self.lay.setContentsMargins(18, 4, 18, 18)
        self.lay.setSpacing(4)
        externe.addWidget(zone_defilante(contenu))

        tete = QHBoxLayout()
        tete.addWidget(section("Cuisines"))
        tete.addStretch(1)
        for libelle, etat in (("Tout", True), ("Aucune", False)):
            b = QPushButton(libelle)
            b.setProperty("role", "lien")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, e=etat: self.tout_cocher(e))
            tete.addWidget(b, 0, Qt.AlignmentFlag.AlignBottom)
        self.lay.addLayout(tete)
        self.grille_puces = Flux(6)
        self.lay.addLayout(self.grille_puces)
        self.puces = []

        self.lay.addWidget(section("Note minimale"))
        ligne = QHBoxLayout()
        self.curseur = QSlider(Qt.Orientation.Horizontal)
        self.curseur.setRange(0, 50)
        self.curseur.setValue(0)
        self.curseur.setAccessibleName("Note minimale")
        self.valeur_note = etiquette("toutes", "valeur")
        self.valeur_note.setFixedWidth(52)
        self.valeur_note.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.curseur.valueChanged.connect(self._note_changee)
        ligne.addWidget(self.curseur, 1)
        ligne.addWidget(self.valeur_note)
        self.lay.addLayout(ligne)

        self.lay.addWidget(section("Prix"))
        ligne = QHBoxLayout()
        ligne.setSpacing(6)
        self.prix = {}
        for k in range(1, 5):
            b = QPushButton("€" * k)
            b.setProperty("role", "bascule")
            b.setAccessibleName(f"Prix {k} sur 4")
            b.setToolTip(f"Prix {k} sur 4")
            b.setCheckable(True)
            b.setChecked(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.toggled.connect(lambda _=False: self.filtres.emit())
            ligne.addWidget(b)
            self.prix[k] = b
        self.lay.addLayout(ligne)

        self.lay.addWidget(section("Affichage"))
        self.options = {}
        for cle, libelle, touche, coche in (("voronoi", "Diagramme de Voronoï", "V", True),
                                            ("delaunay", "Triangulation de Delaunay", "D", False),
                                            ("noms", "Noms des restaurants", "N", True),
                                            ("fleur", "Cercles vides", "F", True),
                                            ("trajet", "Trajet et frontières", "T", True)):
            w = ligne_option(libelle, touche, coche, lambda etat, c=cle: self.calque.emit(c, etat))
            self.options[cle] = w.interrupteur
            self.lay.addWidget(w)

        self.lay.addWidget(section("Couleur des cellules"))
        self.segments_mode = Segments([("mosaique", "Mosaïque"), ("cuisine", "Cuisine"), ("note", "Note")])
        self.segments_mode.change.connect(self.mode.emit)
        self.lay.addWidget(self.segments_mode)

        self.lay.addWidget(section("Données"))
        self.segments_donnees = Segments([("fichier", "Fichier"), ("1000", "1 000"), ("10000", "10 000")])
        self.segments_donnees.change.connect(self.donnees.emit)
        self.lay.addWidget(self.segments_donnees)
        ouvrir = QPushButton("Ouvrir un CSV…")
        ouvrir.setCursor(Qt.CursorShape.PointingHandCursor)
        ouvrir.clicked.connect(lambda: self.donnees.emit("ouvrir"))
        self.lay.addSpacing(6)
        self.lay.addWidget(ouvrir)
        self.resume = etiquette("", "discret")
        self.resume.setWordWrap(True)
        self.resume.setContentsMargins(0, 10, 0, 0)
        self.lay.addWidget(self.resume)
        self.lay.addStretch(1)

    def construire_puces(self, restaurants):
        for p in self.puces:
            p.setParent(None)
        comptes = {}
        for r in restaurants:
            comptes[r.cuisine] = comptes.get(r.cuisine, 0) + 1
        ordre_famille = {nom: k for k, (nom, _, _) in enumerate(FAMILLES)}
        cuisines = sorted(comptes, key=lambda c: (ordre_famille.get(famille(c)[0], len(FAMILLES)), -comptes[c], c))
        self.puces = []
        for c in cuisines:
            puce = Puce(c, famille(c)[1], comptes[c])
            puce.toggled.connect(lambda _=False: self.filtres.emit())
            self.grille_puces.addWidget(puce)
            self.puces.append(puce)

    def tout_cocher(self, etat):
        for p in self.puces:
            p.blockSignals(True)
            p.setChecked(etat)
            p.blockSignals(False)
        self.filtres.emit()

    def reinitialiser(self):
        for p in self.puces:
            p.blockSignals(True)
            p.setChecked(True)
            p.blockSignals(False)
        for b in self.prix.values():
            b.blockSignals(True)
            b.setChecked(True)
            b.blockSignals(False)
        self.curseur.setValue(0)
        self.filtres.emit()

    def _note_changee(self, v):
        self.valeur_note.setText("toutes" if v == 0 else f"≥ {nombre(v / 10, 1)}")
        self.filtres.emit()

    def accepte(self, r):
        actives = {p.cuisine for p in self.puces if p.isChecked()}
        if r.cuisine not in actives:
            return False
        seuil = self.curseur.value() / 10
        if seuil > 0 and (r.note is None or r.note < seuil):
            return False
        return not r.prix or self.prix[r.prix].isChecked()


# --- Panneau de droite : résultats et explications ---------------------------------------

class PanneauResultats(QFrame):
    survol_alternative = Signal(object)
    aller_a = Signal(int)
    comparer = Signal()
    tri_change = Signal(str)
    reinitialiser = Signal()
    camera = Signal(str)
    rejouer = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("resultats")
        self.setFixedWidth(372)
        externe = QVBoxLayout(self)
        externe.setContentsMargins(0, 0, 0, 0)
        self.pile = QStackedWidget()
        externe.addWidget(self.pile)
        self.tri = "distance"

        # Page « vide » : avant le premier clic.
        vide = QWidget()
        lay = QVBoxLayout(vide)
        lay.setContentsMargins(24, 28, 24, 24)
        lay.addWidget(etiquette("Choisissez un point sur la carte", "titre"))
        texte = etiquette("Cliquez n'importe où : la cellule de Voronoï qui contient le clic donne le "
                          "restaurant le plus proche, ses voisins de Delaunay deviennent les alternatives. "
                          "Gardez le bouton enfoncé pour vous déplacer.", "texte")
        texte.setWordWrap(True)
        lay.addSpacing(6)
        lay.addWidget(texte)
        lay.addSpacing(18)
        self.resume_vide = etiquette("", "discret")
        self.resume_vide.setWordWrap(True)
        lay.addWidget(self.resume_vide)
        lay.addSpacing(18)
        lay.addWidget(etiquette("Raccourcis", "section"))
        for touche, action in (("Clic", "placer votre position"), ("Glisser", "se déplacer"),
                               ("Flèches", "se déplacer au clavier (Maj pour aller vite)"),
                               ("Molette", "zoomer"), ("Clic droit", "déplacer la carte"),
                               ("1  2  3", "changer de vue"), ("Échap", "effacer"), ("F1", "aide")):
            ligne = QHBoxLayout()
            ligne.addWidget(etiquette(touche, "touche"))
            ligne.addSpacing(10)
            ligne.addWidget(etiquette(action, "discret"), 1)
            lay.addLayout(ligne)
        lay.addStretch(1)
        self.pile.addWidget(vide)

        # Page « aucun restaurant » : les filtres ont tout exclu.
        aucun = QWidget()
        lay = QVBoxLayout(aucun)
        lay.setContentsMargins(24, 28, 24, 24)
        lay.addWidget(etiquette("Aucun restaurant ne correspond", "titre"))
        t = etiquette("Les filtres actuels excluent tous les restaurants. Élargissez la note, les prix ou "
                      "les cuisines.", "texte")
        t.setWordWrap(True)
        lay.addWidget(t)
        lay.addSpacing(14)
        b = QPushButton("Réinitialiser les filtres")
        b.setProperty("role", "primaire")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(self.reinitialiser.emit)
        lay.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addStretch(1)
        self.pile.addWidget(aucun)

        # Page « résultats » : reconstruite à chaque recherche.
        self.contenu = QWidget()
        self.lay_res = QVBoxLayout(self.contenu)
        self.lay_res.setContentsMargins(20, 18, 20, 20)
        self.lay_res.setSpacing(8)
        self.pile.addWidget(zone_defilante(self.contenu))

        self.page_cones = self._page_explication(
            "Cônes de distance",
            ["Chaque restaurant porte un cône : on descend à mesure qu'on s'en éloigne.",
             "Vu de dessus, le cône le plus haut en un point est celui du restaurant le plus proche. "
             "On retrouve exactement le diagramme de Voronoï, et les crêtes blanches sont ses frontières.",
             "Le contour lavande suit la cellule du restaurant le plus proche de votre position."],
            "Idée de Hoff et al. (SIGGRAPH 1999) : calculer un Voronoï avec une carte graphique.",
            rejouer=False)
        self.page_para = self._page_explication(
            "Relèvement sur un paraboloïde",
            ["Chaque restaurant (x, y) monte à la hauteur x² + y² : les points se posent sur une coupe.",
             "Les faces du dessous de leur enveloppe convexe, projetées à plat, donnent exactement "
             "la triangulation de Delaunay.",
             "Le cercle lavande est le cercle circonscrit d'un triangle. Relevé (en blanc), il reste dans le "
             "plan de la face : un restaurant est dans le cercle si et seulement s'il passe sous la face. "
             "Aucun ne l'est, c'est le critère du cercle vide."],
            "Le triangle montré est celui qui contient votre position.",
            rejouer=True)
        self.note_para = self.page_para.note
        self.pile.addWidget(self.page_cones)
        self.pile.addWidget(self.page_para)

    def _page_explication(self, titre, paragraphes, note, rejouer):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(24, 26, 24, 24)
        lay.setSpacing(10)
        lay.addWidget(etiquette(titre, "titre"))
        for texte in paragraphes:
            lbl = etiquette(texte, "texte")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
        lay.addSpacing(8)
        seg = Segments([("dessus", "Vue de dessus"), ("incline", "Vue inclinée")], actif="incline")
        seg.change.connect(self.camera.emit)
        page.segments = seg
        lay.addWidget(seg, 0, Qt.AlignmentFlag.AlignLeft)
        if rejouer:
            b = QPushButton("Rejouer le relèvement")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(self.rejouer.emit)
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addSpacing(6)
        n = etiquette(note, "discret")
        n.setWordWrap(True)
        lay.addWidget(n)
        page.note = n
        lay.addWidget(etiquette("Faites tourner la scène avec la souris, molette pour zoomer.", "discret"))
        lay.addStretch(1)
        return page

    def montrer_vide(self, resume):
        self.resume_vide.setText(resume)
        self.pile.setCurrentIndex(0)

    def montrer_aucun(self):
        self.pile.setCurrentIndex(1)

    def montrer_vue(self, vue):
        page = self.page_cones if vue == "cones" else self.page_para
        page.segments.choisir("incline")
        self.pile.setCurrentWidget(page)

    def montrer(self, x, y, carte, i, voisins):
        _vider(self.lay_res)
        r = carte.restaurants[i]
        d0 = carte.distance(i, x, y)
        tuile = carte.tuile(i)

        tete = QHBoxLayout()
        tete.addWidget(etiquette("Le plus proche", "section"))
        tete.addStretch(1)
        tete.addWidget(etiquette(f"position ({nombre(x)}, {nombre(y)})", "discret"))
        self.lay_res.addLayout(tete)
        details = [("Cellule", f"{len(tuile)} côtés, aire {nombre(aire(tuile))}"),
                   ("Voisins", f"{nombre(len(voisins), 0)} dans la triangulation de Delaunay")]
        principal = CarteResto(i, r, d0, principal=True, details=details)
        self.lay_res.addWidget(principal)

        self.lay_res.addSpacing(12)
        tete = QHBoxLayout()
        tete.addWidget(etiquette(f"Alternatives : {pluriel(len(voisins), 'voisin')}", "section"))
        tete.addStretch(1)
        self.lay_res.addLayout(tete)
        tri = Segments([("distance", "Distance"), ("note", "Note"), ("prix", "Prix")], actif=self.tri)
        tri.change.connect(self._trier)
        self.lay_res.addWidget(tri)
        cle = {"distance": lambda j: carte.distance(j, x, y),
               "note": lambda j: -(carte.restaurants[j].note or 0),
               "prix": lambda j: (carte.restaurants[j].prix or 9, carte.distance(j, x, y))}[self.tri]
        for j in sorted(voisins, key=cle):
            d = carte.distance(j, x, y)
            c = CarteResto(j, carte.restaurants[j], d, ecart=d - d0)
            c.survole.connect(lambda k, entre: self.survol_alternative.emit(k if entre else None))
            c.clique.connect(self.aller_a.emit)
            self.lay_res.addWidget(c)
        if not voisins:
            self.lay_res.addWidget(etiquette("Aucun voisin : un seul restaurant correspond aux filtres.", "discret"))
        self.lay_res.addSpacing(10)
        comparer = QPushButton("Comparer les alternatives")
        comparer.setProperty("role", "primaire")
        comparer.setCursor(Qt.CursorShape.PointingHandCursor)
        comparer.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        comparer.setEnabled(bool(voisins))
        comparer.clicked.connect(self.comparer.emit)
        self.lay_res.addWidget(comparer)
        astuce = etiquette("Survolez une alternative : la frontière de Voronoï qu'elle partage avec le plus "
                           "proche s'allume, perpendiculaire à leur arête de Delaunay.", "discret")
        astuce.setWordWrap(True)
        self.lay_res.addSpacing(4)
        self.lay_res.addWidget(astuce)
        self.lay_res.addStretch(1)
        self.pile.setCurrentIndex(2)

    def _trier(self, tri):
        self.tri = tri
        self.tri_change.emit(tri)


def _vider(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            # Sorti de la mise en page, un widget resterait visible jusqu'à sa destruction différée.
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout():
            _vider(item.layout())


# --- Fenêtres secondaires ----------------------------------------------------------------

class NuageNoteDistance(QWidget):
    """Nuage distance / note des alternatives (extension C) : en haut à gauche, le meilleur compromis."""

    def __init__(self, lignes):
        super().__init__()
        self.lignes = lignes
        self.setMinimumSize(560, 330)
        self.setMouseTracking(True)
        self.positions = []

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g, d, h, b = 52, 24, 18, 44
        largeur, hauteur = self.width() - g - d, self.height() - h - b
        dmax = max(l["distance"] for l in self.lignes) * 1.12 or 1
        notes = [l["note"] for l in self.lignes if l["note"] is not None]
        nmin = math.floor((min(notes) if notes else 3) * 2 - 1) / 2
        nmax = 5.0

        def px(dist, note):
            return QPointF(g + dist / dmax * largeur, h + (nmax - note) / (nmax - nmin) * hauteur)

        p.setPen(QPen(qc("hairline"), 1))
        police = p.font()
        police.setPointSizeF(8.5)
        p.setFont(police)
        pas_d = _pas_rond(dmax / 5)
        v = 0.0
        while v <= dmax:
            x = g + v / dmax * largeur
            p.setPen(QPen(qc("hairline"), 1))
            p.drawLine(QPointF(x, h), QPointF(x, h + hauteur))
            p.setPen(qc("ink3"))
            p.drawText(QRectF(x - 30, h + hauteur + 6, 60, 16), Qt.AlignmentFlag.AlignHCenter, nombre(v))
            v += pas_d
        n = nmin
        while n <= nmax + 1e-9:
            y = h + (nmax - n) / (nmax - nmin) * hauteur
            p.setPen(QPen(qc("hairline"), 1))
            p.drawLine(QPointF(g, y), QPointF(g + largeur, y))
            p.setPen(qc("ink3"))
            p.drawText(QRectF(0, y - 8, g - 10, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       nombre(n, 1))
            n += 0.5
        p.setPen(qc("ink3"))
        p.drawText(QRectF(g, h + hauteur + 24, largeur, 16), Qt.AlignmentFlag.AlignHCenter, "Distance à votre position")
        p.save()
        p.translate(14, h + hauteur / 2)
        p.rotate(-90)
        p.drawText(QRectF(-hauteur / 2, -8, hauteur, 16), Qt.AlignmentFlag.AlignHCenter, "Note")
        p.restore()
        p.setPen(qc("ink3"))
        p.drawText(QRectF(g + 8, h + 4, 260, 16), Qt.AlignmentFlag.AlignLeft, "Plus près et mieux noté")

        self.positions = []
        rectangles = []
        police.setPointSizeF(9)
        p.setFont(police)
        fm = QFontMetrics(police)
        for l in sorted(self.lignes, key=lambda l: l["principal"]):
            if l["note"] is None:
                continue
            c = px(l["distance"], l["note"])
            self.positions.append((c, l))
            rayon = 7 if l["principal"] else 5
            p.setPen(QPen(qc("s1"), 2))  # anneau de la couleur du fond : sépare les points proches
            p.setBrush(qc("accent2") if l["principal"] else qc("ink3"))
            p.drawEllipse(c, rayon, rayon)
            texte = l["nom"]
            w, ht = fm.horizontalAdvance(texte) + 4, fm.height()
            for dx, dy in ((10, -ht / 2), (-10 - w, -ht / 2), (-w / 2, -ht - 9), (-w / 2, 9)):
                r = QRectF(c.x() + dx, c.y() + dy, w, ht)
                if not any(r.intersects(o) for o in rectangles) and self.rect().contains(r.toRect()):
                    rectangles.append(r)
                    p.setPen(qc("ink" if l["principal"] else "ink2"))
                    p.drawText(r, Qt.AlignmentFlag.AlignCenter, texte)
                    break

    def mouseMoveEvent(self, e):
        for c, l in self.positions:
            if (QPointF(e.position()) - c).manhattanLength() < 12:
                from PySide6.QtWidgets import QToolTip
                QToolTip.showText(e.globalPosition().toPoint(),
                                  f"{l['nom']}\n{l['cuisine']}, note {nombre(l['note'], 1)}, distance {nombre(l['distance'])}",
                                  self)
                return


def _pas_rond(v):
    puissance = 10 ** math.floor(math.log10(max(v, 1e-9)))
    for m in (1, 2, 2.5, 5, 10):
        if m * puissance >= v:
            return m * puissance
    return 10 * puissance


class Comparaison(QDialog):
    def __init__(self, parent, lignes):
        super().__init__(parent)
        self.setWindowTitle("Comparer les alternatives")
        self.resize(720, 640)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 22)
        lay.setSpacing(12)
        lay.addWidget(etiquette("Comparer les alternatives", "titre"))
        sous = etiquette("Le plus proche (lavande) et ses voisins de Delaunay : distance à vous et note.", "discret")
        lay.addWidget(sous)
        cadre = QFrame()
        cadre.setProperty("role", "panneau")
        cl = QVBoxLayout(cadre)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.addWidget(NuageNoteDistance(lignes))
        lay.addWidget(cadre, 1)
        grille = QGridLayout()
        grille.setHorizontalSpacing(18)
        grille.setVerticalSpacing(6)
        for k, titre in enumerate(("", "Restaurant", "Cuisine", "Note", "Prix", "Distance")):
            grille.addWidget(etiquette(titre, "section"), 0, k)
        for r, l in enumerate(sorted(lignes, key=lambda l: (not l["principal"], l["distance"])), start=1):
            grille.addWidget(Pastille(famille(l["cuisine"])[1]), r, 0)
            nom = etiquette(l["nom"] + ("  (le plus proche)" if l["principal"] else ""), "nom" if l["principal"] else "texte")
            grille.addWidget(nom, r, 1)
            grille.addWidget(etiquette(l["cuisine"], "texte"), r, 2)
            grille.addWidget(etiquette("" if l["note"] is None else nombre(l["note"], 1), "valeur"), r, 3)
            prix = QLabel(html_prix(l["prix"]))
            prix.setTextFormat(Qt.TextFormat.RichText)
            grille.addWidget(prix, r, 4)
            grille.addWidget(etiquette(nombre(l["distance"]), "valeur"), r, 5)
        grille.setColumnStretch(1, 1)
        lay.addLayout(grille)
        fermer = QPushButton("Fermer")
        fermer.clicked.connect(self.accept)
        lay.addWidget(fermer, 0, Qt.AlignmentFlag.AlignRight)


class Aide(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Aide")
        self.resize(560, 560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 22)
        lay.setSpacing(10)
        lay.addWidget(etiquette("Comment ça marche", "titre"))
        for texte in (
                "Le diagramme de Voronoï découpe la carte en cellules : chaque point d'une cellule est plus proche "
                "de son restaurant que de tous les autres. Trouver le plus proche revient à trouver la cellule "
                "qui contient le clic.",
                "La triangulation de Delaunay relie deux restaurants quand leurs cellules se touchent : ce sont les "
                "voisins proposés comme alternatives.",
                "Chaque sommet d'une cellule situé à l'intérieur du cadre est le centre d'un cercle vide passant "
                "par au moins trois restaurants : ce sont les cercles lavande affichés autour du plus proche.",
                "Les vues 3D montrent deux raisons géométriques de ces propriétés : les cônes de distance et le "
                "relèvement sur un paraboloïde."):
            lbl = etiquette(texte, "texte")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
        lay.addWidget(section("Raccourcis"))
        grille = QGridLayout()
        grille.setVerticalSpacing(6)
        for k, (touche, action) in enumerate((
                ("V  D  N  F  T", "Voronoï, Delaunay, noms, cercles vides, trajet"),
                ("1  2  3", "carte 2D, cônes 3D, paraboloïde 3D"),
                ("Flèches", "déplacer votre position, carte active (Maj pour aller vite)"), ("Molette", "zoomer"),
                ("Clic droit", "déplacer la carte"), ("Échap", "effacer la sélection"),
                ("Ctrl+S", "enregistrer une image de la vue"), ("Ctrl+O", "ouvrir un CSV"))):
            grille.addWidget(etiquette(touche, "touche"), k, 0)
            grille.addWidget(etiquette(action, "discret"), k, 1)
        grille.setColumnStretch(1, 1)
        lay.addLayout(grille)
        lay.addStretch(1)
        fermer = QPushButton("Fermer")
        fermer.clicked.connect(self.accept)
        lay.addWidget(fermer, 0, Qt.AlignmentFlag.AlignRight)


class Bulle(QLabel):
    """Infobulle ou notification : fenêtre flottante, car la vue VTK dessine en natif."""

    def __init__(self, nom):
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setObjectName(nom)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.minuterie = QTimer(self)
        self.minuterie.setSingleShot(True)
        self.minuterie.timeout.connect(self._estomper)
        self.fondu = QPropertyAnimation(self, b"windowOpacity", self)
        self.fondu.finished.connect(lambda: self.hide() if self.windowOpacity() < 0.05 else None)

    def montrer(self, html, position, duree_ms=None):
        self.fondu.stop()
        self.setWindowOpacity(1.0)
        self.setText(html)
        self.adjustSize()
        self.move(position)
        self.show()
        self.raise_()
        if duree_ms:
            self.minuterie.start(duree_ms)

    def _estomper(self):
        self.fondu.setDuration(260)
        self.fondu.setStartValue(1.0)
        self.fondu.setEndValue(0.0)
        self.fondu.start()


# --- Fenêtre principale ------------------------------------------------------------------

class FenetrePrincipale(QMainWindow):
    def __init__(self, restaurants, source="données", hors_ecran=False):
        super().__init__()
        self.setWindowTitle(typo("Restaurants : Voronoï et Delaunay"))
        self.setMinimumSize(1180, 720)
        self.resize(1480, 900)
        self.source = self.nom_fichier = source
        self.fichier = list(restaurants)
        self.tous = list(restaurants)
        self.cadre = geometrie.cadre(self.tous)
        self.carte = None
        self.position = None
        self.vue = "carte"
        self.glisse = False
        self.pan = None
        self.dernier_mouvement = None
        self.temps_recherche_us = None

        racine = QWidget()
        racine.setObjectName("racine")
        self.setCentralWidget(racine)
        colonne = QVBoxLayout(racine)
        colonne.setContentsMargins(0, 0, 0, 0)
        colonne.setSpacing(0)
        colonne.addWidget(self._barre())
        corps = QHBoxLayout()
        corps.setSpacing(0)
        self.explorer = PanneauExplorer()
        corps.addWidget(self.explorer)
        # Cadre d'un pixel autour de la vue : il s'allume quand la carte reçoit le focus au clavier.
        self.cadre_vue = QFrame()
        self.cadre_vue.setObjectName("cadre_vue")
        lay_vue = QVBoxLayout(self.cadre_vue)
        lay_vue.setContentsMargins(1, 1, 1, 1)
        self.vtk = QVTKRenderWindowInteractor(self.cadre_vue)
        self.vtk.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.vtk.setAccessibleName("Carte des restaurants : flèches pour déplacer votre position")
        self.vtk.installEventFilter(self)
        lay_vue.addWidget(self.vtk)
        corps.addWidget(self.cadre_vue, 1)
        self.resultats = PanneauResultats()
        corps.addWidget(self.resultats)
        colonne.addLayout(corps, 1)
        self._barre_etat()

        self.rw = self.vtk.GetRenderWindow()
        if hors_ecran:
            self.rw.SetOffScreenRendering(True)
        self.rw.SetMultiSamples(8)
        self.carte2d = Carte2D()
        self.cones = VueCones()
        self.para = VueParaboloide()
        self.a_jour = {"cones": False, "paraboloide": False}
        self.rw.AddRenderer(self.carte2d.ren)
        self.iren = self.rw.GetInteractor()
        self.style_2d = vtk.vtkInteractorStyleUser()
        self.style_3d = vtk.vtkInteractorStyleTrackballCamera()
        for style in (self.style_2d, self.style_3d):
            # Neutralise les touches VTK par défaut (« e » ou « q » fermeraient la vue, etc.).
            style.AddObserver("CharEvent", lambda *_: None)
        self.iren.SetInteractorStyle(self.style_2d)
        for evenement, rappel in (("LeftButtonPressEvent", self._appui), ("LeftButtonReleaseEvent", self._relache),
                                  ("MouseMoveEvent", self._mouvement), ("RightButtonPressEvent", self._pan_debut),
                                  ("RightButtonReleaseEvent", self._pan_fin),
                                  ("MiddleButtonPressEvent", self._pan_debut),
                                  ("MiddleButtonReleaseEvent", self._pan_fin),
                                  ("MouseWheelForwardEvent", lambda *_: self._molette(1 / 1.2)),
                                  ("MouseWheelBackwardEvent", lambda *_: self._molette(1.2)),
                                  ("LeaveEvent", self._sortie), ("ConfigureEvent", self._redimension),
                                  ("KeyPressEvent", self._clavier)):
            self.iren.AddObserver(evenement, rappel)
        self.animateur = Animateur(self.rendre)
        self.infobulle = Bulle("infobulle")
        self.toast = Bulle("toast")
        if hors_ecran:  # captures et tests : aucune fenêtre ne doit apparaître à l'écran
            for bulle in (self.infobulle, self.toast):
                bulle.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        self.survol_minuterie = QTimer(self)
        self.survol_minuterie.setSingleShot(True)
        self.survol_minuterie.timeout.connect(self._traiter_mouvement)
        self.minuterie_panneau = QTimer(self)
        self.minuterie_panneau.setSingleShot(True)
        self.minuterie_panneau.timeout.connect(self._rafraichir_resultats)
        self.dernier_plus_proche = None

        self.explorer.filtres.connect(self.reconstruire)
        self.explorer.calque.connect(self._calque)
        self.explorer.mode.connect(self._mode)
        self.explorer.donnees.connect(self._donnees)
        self.resultats.survol_alternative.connect(self._survol_alternative)
        self.resultats.aller_a.connect(self.aller_vers_restaurant)
        self.resultats.comparer.connect(self._comparer)
        self.resultats.tri_change.connect(lambda _: self._rafraichir_resultats())
        self.resultats.reinitialiser.connect(self.explorer.reinitialiser)
        self.resultats.camera.connect(self._camera_3d)
        self.resultats.rejouer.connect(self._lever)
        self._raccourcis()

        self.explorer.construire_puces(self.tous)
        self.reconstruire()
        # Contour de focus au clavier seulement : un clic sur un bouton ne lui donne pas le focus.
        for bouton in self.findChildren(QPushButton):
            bouton.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.vtk.setFocus()  # la carte écoute les flèches dès l'ouverture

    # -- construction de l'interface ----------------------------------------------------

    def _barre(self):
        barre = QFrame()
        barre.setObjectName("barre")
        barre.setFixedHeight(56)
        lay = QHBoxLayout(barre)
        lay.setContentsMargins(18, 0, 14, 0)
        lay.setSpacing(10)
        lay.addWidget(Marque())
        marque = QLabel("Restaurants")
        marque.setObjectName("marque")
        lay.addWidget(marque)
        sous = QLabel("Voronoï et Delaunay")
        sous.setObjectName("sousmarque")
        lay.addWidget(sous)
        lay.addStretch(1)
        self.segments_vue = Segments([("carte", "Carte 2D"), ("cones", "Cônes 3D"), ("paraboloide", "Paraboloïde 3D")])
        self.segments_vue.change.connect(self.changer_vue)
        lay.addWidget(self.segments_vue, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addStretch(1)
        capture = QPushButton(typo("Exporter l'image"))
        capture.setToolTip("Enregistrer une image PNG de la vue, en double résolution (Ctrl+S)")
        capture.setCursor(Qt.CursorShape.PointingHandCursor)
        capture.clicked.connect(self.enregistrer_image)
        lay.addWidget(capture)
        aide = QPushButton("Aide")
        aide.setCursor(Qt.CursorShape.PointingHandCursor)
        aide.clicked.connect(lambda: Aide(self).exec())
        lay.addWidget(aide)
        return barre

    def _barre_etat(self):
        etat = self.statusBar()
        etat.setSizeGripEnabled(False)
        self.etat_compte = QLabel()
        self.etat_temps = QLabel()
        self.etat_position = QLabel()
        etat.addWidget(self.etat_compte)
        etat.addWidget(self.etat_temps, 1)
        etat.addPermanentWidget(self.etat_position)

    def _raccourcis(self):
        for touche, action in (("V", lambda: self._basculer("voronoi")), ("D", lambda: self._basculer("delaunay")),
                               ("N", lambda: self._basculer("noms")), ("F", lambda: self._basculer("fleur")),
                               ("T", lambda: self._basculer("trajet")),
                               ("1", lambda: self.changer_vue("carte")), ("2", lambda: self.changer_vue("cones")),
                               ("3", lambda: self.changer_vue("paraboloide")), ("Escape", self.effacer),
                               ("Ctrl+S", self.enregistrer_image), ("Ctrl+O", lambda: self._donnees("ouvrir")),
                               ("F1", lambda: Aide(self).exec())):
            QShortcut(QKeySequence(touche), self, activated=action)

    # -- données et filtres -------------------------------------------------------------

    def reconstruire(self):
        """Filtre, reconstruit la Carte (Voronoï + Delaunay) dans le cadre fixe, puis la sélection."""
        garde = [r for r in self.tous if self.explorer.accepte(r)]
        self.animateur.arreter("aller")
        self.minuterie_panneau.stop()
        # Nouvelle Carte = nouveaux indices : le panneau doit être reconstruit tout de suite.
        self.dernier_plus_proche = None
        etendue = max(self.cadre[1] - self.cadre[0], self.cadre[3] - self.cadre[2])
        PRECISION[0] = 0 if etendue >= 100 else 1 if etendue >= 10 else 2 if etendue >= 1 else 3
        self.carte = geometrie.Carte(garde, bornes=self.cadre)
        self.carte2d.charger(self.carte, self.cadre)
        self.a_jour = {"cones": False, "paraboloide": False}
        taille = self.rw.GetSize()
        self.carte2d.cadrer(max(taille[0], 1), max(taille[1], 1))
        self._etat()
        notes = [r.note for r in self.tous if r.note is not None]
        resume = (f"{pluriel(len(self.tous), 'restaurant')}, {pluriel(len({r.cuisine for r in self.tous}), 'cuisine')}"
                  + (f", note moyenne {nombre(sum(notes) / len(notes), 2)}" if notes else "") + f". Source : {self.source}.")
        resume = typo(resume)
        self.explorer.resume.setText(resume)
        if self.carte.n == 0:
            self.carte2d.effacer_selection()
            if self.vue == "carte":
                self.resultats.montrer_aucun()
        elif self.position is not None:
            self.choisir(*self.position, animer=False)
        elif self.vue == "carte":
            self.resultats.montrer_vide(resume)
        if self.vue != "carte":
            self._preparer_3d(self.vue)
        self.rendre()

    def _etat(self):
        t = self.carte.temps_ms
        self.etat_compte.setText(f"{nombre(self.carte.n, 0)} affichés sur {nombre(len(self.tous), 0)}")
        recherche = f"   Recherche {nombre(self.temps_recherche_us, 1)} µs" if self.temps_recherche_us else ""
        self.etat_temps.setText(f"Voronoï {nombre(t.get('voronoi', 0), 1)} ms   "
                                f"Delaunay {nombre(t.get('delaunay', 0), 1)} ms{recherche}")

    def _donnees(self, choix):
        if choix == "ouvrir":
            chemin, _ = QFileDialog.getOpenFileName(self, "Ouvrir un fichier de restaurants", str(Path.cwd()),
                                                    "Fichiers CSV (*.csv)")
            if not chemin:
                return
            try:
                restaurants = geometrie.charger_csv(chemin)
            except geometrie.ErreurDonnees as erreur:
                QMessageBox.warning(self, "Fichier refusé", f"{erreur}\n\n{CONSEIL_CSV}")
                return
            self.fichier, self.nom_fichier = restaurants, Path(chemin).name
            choix = "fichier"
            self.explorer.segments_donnees.choisir("fichier")
        if choix == "fichier":
            self.tous, self.source = list(self.fichier), self.nom_fichier
            self.cadre = geometrie.cadre(self.tous)
        else:
            # Extension D : même zone que la carte du fichier, beaucoup plus de restaurants.
            zone = geometrie.cadre(self.fichier)
            self.tous = geometrie.generer_restaurants(int(choix), graine=7, bornes=zone)
            self.source = f"{nombre(len(self.tous), 0)} restaurants aléatoires"
            self.cadre = geometrie.cadre(self.tous, marge=0.01)
        self.animateur.arreter("aller")
        self.position = None
        self.carte2d.trajet_effacer()
        self.explorer.construire_puces(self.tous)
        self.reconstruire()

    # -- recherche ----------------------------------------------------------------------

    def choisir(self, x, y, animer=True):
        """Le cœur du sujet : cellule de Voronoï contenant (x, y), puis voisins de Delaunay."""
        self.position = (x, y)
        debut = time.perf_counter_ns()
        i = self.carte.plus_proche(x, y)
        self.temps_recherche_us = (time.perf_counter_ns() - debut) / 1000
        if i is None:
            return
        voisins = self.carte.voisins(i)
        self.carte2d.selection(x, y, i, voisins)
        if animer:
            self.animateur.lancer("selection", 0.6, self.carte2d.animer_selection)
            self.animateur.lancer("pulse", 1.0, self.carte2d.pulse, courbe=lambda t: t)
        if self.vue != "carte":
            pass  # le panneau de droite explique la vue 3D : on ne le remplace pas
        elif animer or i != self.dernier_plus_proche:
            self.minuterie_panneau.stop()
            self.resultats.montrer(x, y, self.carte, i, voisins)
        elif not self.minuterie_panneau.isActive():
            # Pendant un glisser, seules les distances changent : panneau rafraîchi au plus toutes les 90 ms.
            self.minuterie_panneau.start(90)
        self.dernier_plus_proche = i
        self.etat_position.setText(f"x {nombre(x)}   y {nombre(y)}")
        self._etat()

    def _rafraichir_resultats(self):
        if self.vue == "carte" and self.position is not None and self.carte.n:
            x, y = self.position
            i = self.carte.plus_proche(x, y)
            self.resultats.montrer(x, y, self.carte, i, self.carte.voisins(i))

    def effacer(self):
        self.animateur.arreter("aller")
        self.position = None
        self.dernier_plus_proche = None
        self.carte2d.effacer_selection()
        self.carte2d.trajet_effacer()
        if self.vue == "carte":
            if self.carte.n:
                self.resultats.montrer_vide(self.explorer.resume.text())
            else:
                self.resultats.montrer_aucun()
        self.cones.montrer(None)
        self.rendre()

    def aller_vers_restaurant(self, indice):
        """Glisse la position jusqu'au restaurant choisi (extension C), frontières comprises."""
        if self.position is None:
            return
        x0, y0 = self.position
        x1, y1 = self.carte.sites[indice]
        self.carte2d.trajet_effacer()
        self.carte2d.trajet_ajouter(x0, y0)
        precedent = [x0, y0]

        def pas(t):
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            self._deplacer_vers(precedent[0], precedent[1], x, y)
            precedent[:] = [x, y]

        self.animateur.lancer("aller", 0.7, pas, courbe=douce)

    def _deplacer_vers(self, xa, ya, xb, yb):
        """Un pas de déplacement (extension A) : franchissements exacts, trajet, notification."""
        avant = self.dernier_plus_proche
        croisements = self.carte.traversees(xa, ya, xb, yb)
        apres = self.carte.plus_proche(xb, yb)
        if not croisements and avant is not None and apres != avant and avant < self.carte.n:
            # Un pas qui s'arrête pile sur une frontière ne la « traverse » pas strictement :
            # sans ce rattrapage, le changement de restaurant passerait inaperçu au pas suivant.
            croisements = [(*self._point_frontiere(avant, apres, xa, ya, xb, yb), avant, apres)]
        self.carte2d.trajet_ajouter(xb, yb, croisements)
        self.choisir(xb, yb, animer=False)
        if croisements:
            chaine = [croisements[0][2]] + [c[3] for c in croisements]
            noms = " → ".join(f"<b>{self.carte.restaurants[k].nom}</b>" for k in chaine)
            titre = "Frontière franchie" if len(croisements) == 1 else f"{len(croisements)} frontières franchies"
            self.notifier(f"{titre} : {noms}")

    def _point_frontiere(self, a, b, xa, ya, xb, yb):
        """Point du segment où les restaurants a et b sont à égale distance (médiatrice)."""
        (sax, say), (sbx, sby) = self.carte.sites[a], self.carte.sites[b]

        def ecart(x, y):
            return (x - sax) ** 2 + (y - say) ** 2 - (x - sbx) ** 2 - (y - sby) ** 2

        f0, f1 = ecart(xa, ya), ecart(xb, yb)  # l'écart varie linéairement le long du segment
        t = 1.0 if f0 == f1 else min(1.0, max(0.0, f0 / (f0 - f1)))
        return xa + (xb - xa) * t, ya + (yb - ya) * t

    def notifier(self, html):
        coin = self.vtk.mapToGlobal(QPoint(self.vtk.width() // 2, self.vtk.height() - 64))
        self.toast.setText(html)
        self.toast.adjustSize()
        self.toast.montrer(html, coin - QPoint(self.toast.width() // 2, 0), 1800)

    # -- souris (carte 2D) --------------------------------------------------------------

    def _position_monde(self):
        sx, sy = self.iren.GetEventPosition()
        return self.carte2d.ecran_vers_monde(sx, sy)

    def _appui(self, *_):
        if self.vue != "carte" or self.carte.n == 0:
            return
        self.infobulle.hide()
        x, y = self._position_monde()
        self.glisse = True
        self.animateur.arreter("aller")
        self.carte2d.trajet_effacer()
        self.carte2d.trajet_ajouter(x, y)
        self.choisir(x, y)
        self.rendre()

    def _relache(self, *_):
        self.glisse = False

    def _pan_debut(self, *_):
        if self.vue == "carte":
            self.pan = self.iren.GetEventPosition()

    def _pan_fin(self, *_):
        self.pan = None

    def _mouvement(self, *_):
        if self.vue != "carte":
            return
        self.dernier_mouvement = self.iren.GetEventPosition()
        if not self.survol_minuterie.isActive():
            self.survol_minuterie.start(0)  # regroupe les mouvements : un rendu par passage de boucle

    def _traiter_mouvement(self):
        if self.dernier_mouvement is None or self.carte is None or self.vue != "carte":
            return
        sx, sy = self.dernier_mouvement
        if self.pan is not None:
            self.carte2d.deplacer(sx - self.pan[0], sy - self.pan[1])
            self.pan = (sx, sy)
            if self.carte2d.selection_courante:
                self.carte2d.selection(*self.carte2d.selection_courante)
            self.rendre()
            return
        x, y = self.carte2d.ecran_vers_monde(sx, sy)
        self.etat_position.setText(f"x {nombre(x)}   y {nombre(y)}")
        if self.carte.n == 0:
            return
        if self.glisse and self.position is not None:
            self._deplacer_vers(*self.position, x, y)
            self.rendre()
            return
        i = self.carte.plus_proche(x, y)
        self.carte2d.survol(i)
        r = self.carte.restaurants[i]
        html = (f"<b style='color:{JETONS['ink']}'>{r.nom}</b><br>"
                f"<span style='color:{JETONS['ink3']}'>{r.cuisine}</span>&nbsp;&nbsp;{html_prix(r.prix)}"
                f"&nbsp;&nbsp;{html_note(r.note)}<br>"
                f"<span style='color:{JETONS['ink3']}'>à</span> <span style='color:{JETONS['ink']}'>"
                f"{nombre(self.carte.distance(i, x, y))}</span>")
        self.infobulle.montrer(html, QCursor.pos() + QPoint(16, 18))
        self.rendre()

    def _sortie(self, *_):
        self.infobulle.hide()
        self.carte2d.survol(None)
        self.rendre()

    def _clavier(self, *_):
        """Alternative clavier au clic (carte active) : flèches pour se déplacer, Maj pour aller plus vite."""
        if self.vue != "carte" or self.carte.n == 0:
            return
        touche = self.iren.GetKeySym()
        x0, x1, y0, y1 = self.cadre
        pas = (x1 - x0) / (12 if self.iren.GetShiftKey() else 60)
        direction = {"Left": (-pas, 0), "Right": (pas, 0), "Up": (0, pas), "Down": (0, -pas)}.get(touche)
        if self.position is None and (direction or touche in ("Return", "space")):
            self.carte2d.trajet_effacer()
            self.carte2d.trajet_ajouter((x0 + x1) / 2, (y0 + y1) / 2)
            self.choisir((x0 + x1) / 2, (y0 + y1) / 2)
        elif direction:
            x, y = self.position
            nx = min(max(x + direction[0], x0), x1)
            ny = min(max(y + direction[1], y0), y1)
            self._deplacer_vers(x, y, nx, ny)
        self.rendre()

    def eventFilter(self, objet, evenement):
        # Contour de focus seulement pour le clavier (Tab), pas après un clic : équivalent de :focus-visible.
        if objet is self.vtk and evenement.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            clavier = (evenement.type() == QEvent.Type.FocusIn
                       and evenement.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason))
            self.cadre_vue.setProperty("actif", clavier)
            self.cadre_vue.style().unpolish(self.cadre_vue)
            self.cadre_vue.style().polish(self.cadre_vue)
        return super().eventFilter(objet, evenement)

    def _molette(self, facteur):
        if self.vue != "carte":
            return
        sx, sy = self.iren.GetEventPosition()
        self.carte2d.zoomer(facteur, sx, sy)
        if self.carte2d.selection_courante:
            self.carte2d.selection(*self.carte2d.selection_courante)
        self.rendre()

    def _redimension(self, *_):
        largeur, hauteur = self.rw.GetSize()
        self.carte2d.redimensionner(largeur, hauteur)
        for vue in (self.cones, self.para):
            # Pas de repose de la caméra : on garde la rotation et le zoom choisis à la souris.
            vue.orbite.taille_px = (largeur, hauteur)
        if self.carte2d.selection_courante:
            self.carte2d.selection(*self.carte2d.selection_courante)

    # -- calques, couleurs ----------------------------------------------------------------

    def _calque(self, nom, visible):
        self.carte2d.calque(nom, visible)
        self.rendre()

    def _basculer(self, nom):
        inter = self.explorer.options[nom]
        inter.setChecked(not inter.isChecked())

    def _mode(self, mode):
        self.carte2d.appliquer_mode(mode)
        self.rendre()

    def _survol_alternative(self, j):
        if self.position is None:
            return
        i = self.carte.plus_proche(*self.position)
        self.carte2d.dual(i, j)
        self.rendre()

    def _comparer(self):
        if self.position is None:
            return
        x, y = self.position
        i = self.carte.plus_proche(x, y)
        lignes = []
        for j in [i] + self.carte.voisins(i):
            r = self.carte.restaurants[j]
            lignes.append({"nom": r.nom, "cuisine": r.cuisine, "note": r.note, "prix": r.prix,
                           "distance": self.carte.distance(j, x, y), "principal": j == i})
        Comparaison(self, lignes).exec()

    # -- vues -----------------------------------------------------------------------------

    def changer_vue(self, vue):
        if vue == self.vue:
            return
        anciens = {"carte": self.carte2d.ren, "cones": self.cones.ren, "paraboloide": self.para.ren}
        self.rw.RemoveRenderer(anciens[self.vue])
        self.vue = vue
        self.segments_vue.choisir(vue)
        self.survol_minuterie.stop()  # un survol de la carte en attente ne doit pas surgir en 3D
        self.dernier_mouvement = None
        self.glisse = False
        self.pan = None
        self.infobulle.hide()
        self.carte2d.survol(None)
        if vue == "carte":
            self.rw.AddRenderer(self.carte2d.ren)
            self.iren.SetInteractorStyle(self.style_2d)
            if self.carte.n == 0:
                self.resultats.montrer_aucun()
            elif self.position is not None:
                self._rafraichir_resultats()
            else:
                self.resultats.montrer_vide(self.explorer.resume.text())
        else:
            self._preparer_3d(vue)
            self.rw.AddRenderer(anciens[vue])
            self.iren.SetInteractorStyle(self.style_3d)
            self.resultats.montrer_vue(vue)
            scene = self.cones if vue == "cones" else self.para
            self.animateur.lancer("camera", 1.6, scene.orbite.poser, courbe=douce)
            if vue == "paraboloide":
                self._lever()
        self.rendre()

    def _preparer_3d(self, vue):
        taille = self.rw.GetSize()
        if vue == "cones":
            if not self.a_jour["cones"]:
                self.cones.charger(self.carte, self.cadre, taille)
                self.a_jour["cones"] = True
            self.cones.montrer(self.carte2d.selection_courante)
        else:
            if not self.a_jour["paraboloide"]:
                self.para.charger(self.carte, self.cadre, taille)
                self.a_jour["paraboloide"] = True
            contient = self.para.exemple(self.position)
            if self.position is None:
                note = "Placez votre position sur la carte 2D : le triangle qui la contient sera montré ici."
            elif contient:
                note = "Le triangle montré est celui qui contient votre position."
            else:
                note = "Votre position est hors de la triangulation : le triangle montré est un exemple."
            self.resultats.note_para.setText(typo(note))

    def _camera_3d(self, choix):
        scene = self.cones if self.vue == "cones" else self.para
        depart, arrivee = scene.orbite.t, (0.0 if choix == "dessus" else 1.0)
        self.animateur.lancer("camera", 1.2, lambda t: scene.orbite.poser(depart + (arrivee - depart) * t),
                              courbe=douce)

    def _lever(self):
        self.animateur.lancer("lever", 1.8, self.para.lever, courbe=douce)

    # -- rendu, captures --------------------------------------------------------------------

    def rendre(self):
        self.rw.Render()

    def image_vue(self, echelle=1):
        """Image de la vue VTK (pour la capture et la composition des captures d'écran)."""
        filtre = vtk.vtkWindowToImageFilter()
        filtre.SetInput(self.rw)
        filtre.SetScale(echelle)
        filtre.ReadFrontBufferOff()
        filtre.Update()
        image = filtre.GetOutput()
        largeur, hauteur, _ = image.GetDimensions()
        pixels = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(hauteur, largeur, -1)[::-1].copy()
        format_qt = QImage.Format.Format_RGBA8888 if pixels.shape[2] == 4 else QImage.Format.Format_RGB888
        return QImage(pixels.data, largeur, hauteur, pixels.strides[0], format_qt).copy()

    def enregistrer_image(self):
        chemin, _ = QFileDialog.getSaveFileName(self, "Enregistrer une image de la vue",
                                                str(Path.cwd() / f"vue_{self.vue}.png"), "Images PNG (*.png)")
        if chemin:
            self.image_vue(2).save(chemin)
            self.notifier(f"Image enregistrée : <b>{Path(chemin).name}</b>")

    def capture_complete(self, chemin):
        """Capture de toute la fenêtre : widgets Qt + image VTK collée à sa place."""
        QApplication.processEvents()  # widgets ajoutés et animations en attente d'abord
        self.rendre()
        fond = self.grab().toImage()
        vue = self.image_vue()
        p = QPainter(fond)
        coin = self.vtk.mapTo(self, QPoint(0, 0))
        p.drawImage(QRectF(coin.x(), coin.y(), self.vtk.width(), self.vtk.height()), vue)
        p.end()
        fond.save(str(chemin))

    def closeEvent(self, e):
        # Arrêter d'abord tout ce qui peut encore demander un rendu : après Finalize(),
        # un Render() provoque une erreur OpenGL et peut bloquer la fermeture.
        self.animateur.anims.clear()
        for minuterie in (self.animateur.minuterie, self.survol_minuterie, self.minuterie_panneau):
            minuterie.stop()
        self.vtk.removeEventFilter(self)  # la destruction envoie encore des événements de focus
        self.infobulle.close()
        self.toast.close()
        self.vtk.Finalize()
        super().closeEvent(e)


def appliquer_theme(app):
    if THEME.exists():
        app.setStyleSheet(THEME.read_text(encoding="utf-8"))
