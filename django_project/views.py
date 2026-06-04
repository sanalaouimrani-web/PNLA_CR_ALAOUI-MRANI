# -*- coding: utf-8 -*-
"""
Vues Django pour le laboratoire d'optimisation numérique (Master IARO).
Faculté des Sciences de Meknès.
"""

from django.shortcuts import render
from django.views import View
import matplotlib
matplotlib.use('Agg') # Mode non-interactif pour serveur headless
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from .optimization import minimiser_fonction, evaluer_fonction_complete

class OptimizationLabView(View):
    # Fonctions de TD prédéfinies pour le laboratoire
    PRELOADED_FUNCTIONS = {
        '1': {
            'title': "Fonction 1 (TD3 Ex 1&2)",
            'formula': "x1**2 + x1*x2 + x2**2",
            'x0': [1.0, 1.0],
            'td': "TD3 Exercices 1 & 2",
            'desc': "Fonction quadratique couplée. Permet d'observer l'effet de cisaillement et le zigzag du gradient descendant lié au conditionnement spectral."
        },
        '2': {
            'title': "Fonction 2 (TD3 Ex 3&4)",
            'formula': "x1**2 + x2**2",
            'x0': [2.0, 2.0],
            'td': "TD3 Exercices 3 & 4",
            'desc': "Cuvette sphérique parfaite à gradient radial. Idéale : Newton ou gradient à pas optimal y convergent en une seule itération."
        },
        '3': {
            'title': "Fonction 3 (TD4 Ex 1&2)",
            'formula': "(x1 - 1)**2 + 2*x2**2",
            'x0': [0.0, 0.0],
            'td': "TD4 Exercices 1 & 2",
            'desc': "Quadratique décalée avec anisotropie d'échelonnage. Présente un zigzag prononcé si le pas de pente n'est pas optimal."
        },
        '4': {
            'title': "Fonction 4 (TD4 Ex 3&4)",
            'formula': "5*x1**2 + 6*x2**2 + 2*x1*x2 + 9*x1 + 5*x2",
            'x0': [0.0, 0.0],
            'td': "TD4 Exercices 3 & 4",
            'desc': "Forme quadratique complexe issue de l'exercice d'examen. Minimum global excentré, idéal pour comparer avec BFGS et Newton."
        }
    }

    def get(self, request):
        # Affichage initial de la page avec la Fonction 1 pré-sélectionnée
        context = {
            'preloaded_functions': self.PRELOADED_FUNCTIONS,
            'selected_id': '1',
            'formula': self.PRELOADED_FUNCTIONS['1']['formula'],
            'x0_x1': self.PRELOADED_FUNCTIONS['1']['x0'][0],
            'x0_x2': self.PRELOADED_FUNCTIONS['1']['x0'][1],
            'max_iter': 50,
            'epsilon': 0.0001,
            'direction_method': 'gradient',
            'step_method': 'optimal',
            'fixed_alpha': 0.1,
            'show_results': False
        }
        return render(request, 'lab_optimization/template.html', context)

    def post(self, request):
        # Récupération sécurisée des configurations depuis le formulaire HTML
        function_id = request.POST.get('function_id', 'custom')
        formula = request.POST.get('formula', '').strip()
        
        try:
            x0_x1 = float(request.POST.get('x0_x1', 0.0))
            x0_x2 = float(request.POST.get('x0_x2', 0.0))
            max_iter = int(request.POST.get('max_iter', 50))
            epsilon = float(request.POST.get('epsilon', 1e-4))
            fixed_alpha = float(request.POST.get('fixed_alpha', 0.1))
        except ValueError:
            x0_x1, x0_x2, max_iter, epsilon, fixed_alpha = 0.0, 0.0, 50, 1e-4, 0.1

        direction_method = request.POST.get('direction_method', 'gradient')
        step_method = request.POST.get('step_method', 'optimal')

        # Si l'utilisateur choisit une fonction préchargée
        if function_id in self.PRELOADED_FUNCTIONS:
            selected_func = self.PRELOADED_FUNCTIONS[function_id]
            formula = selected_func['formula']
            # Ne récupérer x0 depuis la carte que si l'utilisateur n'a pas surchargé
            if 'override_x0' not in request.POST:
                x0_x1 = selected_func['x0'][0]
                x0_x2 = selected_func['x0'][1]

        # Resolution numérique de l'optimisation
        error_msg = None
        iterations = []
        chart_base64 = ""

        try:
            iterations = minimiser_fonction(
                formula_str=formula,
                x0=(x0_x1, x0_x2),
                max_iter=max_iter,
                eps=epsilon,
                direction_method=direction_method,
                step_method=step_method,
                fixed_alpha=fixed_alpha
            )
            
            # Génération du graphique de convergence avec Matplotlib
            if iterations:
                chart_base64 = self.generer_graphique_contours(formula, iterations)
        except Exception as e:
            error_msg = f"Erreur lors du calcul d'optimisation : {str(e)}"

        context = {
            'preloaded_functions': self.PRELOADED_FUNCTIONS,
            'selected_id': function_id,
            'formula': formula,
            'x0_x1': x0_x1,
            'x0_x2': x0_x2,
            'max_iter': max_iter,
            'epsilon': epsilon,
            'direction_method': direction_method,
            'step_method': step_method,
            'fixed_alpha': fixed_alpha,
            'iterations': iterations,
            'chart_image': chart_base64,
            'error_message': error_msg,
            'show_results': True
        }
        return render(request, 'lab_optimization/template.html', context)

    def generer_graphique_contours(self, formula_str, iterations):
        """
        Dessine les lignes de niveau 2D de la fonction ainsi que la trajectoire suivie
        par les itérations x^k -> x^{k+1}. Intègre les contours de hauteur f(x).
        """
        # Extraire l'historique des points d'itération
        pts = np.array([pt['x'] for pt in iterations])
        
        # Bornes intelligentes basées sur la trajectoire pour cadrer le tracé
        margin = 1.0
        x1_min, x1_max = pts[:, 0].min() - margin, pts[:, 0].max() + margin
        x2_min, x2_max = pts[:, 1].min() - margin, pts[:, 1].max() + margin
        
        # Résolution du maillage de grille rectangulaire 2D
        nx, ny = 120, 120
        grid_x1 = np.linspace(x1_min, x1_max, nx)
        grid_x2 = np.linspace(x2_min, x2_max, ny)
        X1, X2 = np.meshgrid(grid_x1, grid_x2)
        
        # Évaluation de la grille via l'utilitaire d'optimisation
        Z = np.zeros_like(X1)
        # On utilise une lambdification rapide symbolique
        import sympy as sp
        x1, x2 = sp.symbols('x1 x2')
        formula_clean = formula_str.replace('x_1', 'x1').replace('x_2', 'x2').replace('^', '**')
        
        try:
            lamb_f = sp.lambdify((x1, x2), sp.sympify(formula_clean), 'numpy')
            Z = lamb_f(X1, X2)
            # Enlever les valeurs aberrantes ou inf pour affichage stable
            if np.any(np.isinf(Z)) or np.any(np.isnan(Z)):
                raise ValueError("Grille non-bornée détectée")
        except Exception:
            # Fallback point-par-point sécurisé
            for r in range(ny):
                for c in range(nx):
                    try:
                        val, _, _ = evaluer_fonction_complete(formula_str, X1[r, c], X2[r, c])
                        Z[r, c] = val
                    except:
                        Z[r, c] = 0.0

        # Création et configuration du style de la figure Matplotlib
        fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=100)
        fig.patch.set_facecolor('#f8fafc') # fond ardoise clair cool
        ax.set_facecolor('#ffffff')
        
        # Tracé des contours remplis (Scalar Field shading)
        contours_fill = ax.contourf(X1, X2, Z, levels=18, cmap='viridis', alpha=0.82)
        fig.colorbar(contours_fill, ax=ax, label='f(x1, x2)')
        
        # Tracé des lignes de niveau
        contours_line = ax.contour(X1, X2, Z, levels=18, colors='#475569', linewidths=0.6, alpha=0.5)
        ax.clabel(contours_line, inline=True, fontsize=8, fmt='%.1f')
        
        # Tracé de la trajectoire d'itérations
        ax.plot(pts[:, 0], pts[:, 1], color='#e11d48', linestyle='--', linewidth=1.5, marker='o', markersize=4, label='Cheminement ($x^k$)')
        
        # Représentation du point initial (Star gold) et du point final de convergence (Star green)
        ax.plot(pts[0, 0], pts[0, 1], color='#eab308', marker='p', markersize=10, label='$x^0$ (Initial)', markeredgecolor='black')
        ax.plot(pts[-1, 0], pts[-1, 1], color='#22c55e', marker='*', markersize=14, label='$x^*$ (Final)', markeredgecolor='black')
        
        # Dessiner des flèches vectorielles pour visualiser d_k * alpha_k
        for idx in range(len(pts) - 1):
            dx = pts[idx+1, 0] - pts[idx, 0]
            dy = pts[idx+1, 1] - pts[idx, 1]
            ax.annotate('', xy=(pts[idx+1, 0], pts[idx+1, 1]), xytext=(pts[idx, 0], pts[idx, 1]),
                        arrowprops=dict(arrowstyle="-|>", color='#dc2626', lw=1.2, mutation_scale=12, shrinkA=0, shrinkB=0), alpha=0.85)

        ax.set_title("Visualisation d'Optimisation : Carte des Contours et Cheminement", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('$x_1$', fontsize=10)
        ax.set_ylabel('$x_2$', fontsize=10)
        ax.grid(color='#e2e8f0', linestyle=':', linewidth=0.5)
        ax.legend(loc='best', framealpha=0.9, facecolor='#ffffff', edgecolor='#cbd5e1')
        
        # Sauvegarde en buffer mémoire BytesIO et conversion Base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        image_bytes = buf.getvalue()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
