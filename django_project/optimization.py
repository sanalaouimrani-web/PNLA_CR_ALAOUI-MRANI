# -*- coding: utf-8 -*-
"""
Module d'Optimisation Numérique 2D pour le Master IARO.
Université Moulay Ismaïl, Faculté des Sciences de Meknès.

Auteurs: Laboratoire d'Optimisation Numérique IARO
Ce fichier contient la logique de calcul analytique du gradient, de la Hessienne
avec SymPy, et les méthodes de descente (Gradient Pas Fixe/Optimal, Newton, BFGS)
avec choix de recherche linéaire (Armijo, Wolfe, Pas Fixe, Pas Optimal).
"""

import sympy as sp
import numpy as np

def evaluer_fonction_complete(formule_str, x1_val, x2_val):
    """
    Explore l'expression analytique de la fonction via SymPy,
    calcule la valeur, le gradient analytique et la matrice Hessienne.
    """
    x1, x2 = sp.symbols('x1 x2')
    
    # Nettoyage de la formule (s'assurer de supporter x_1, x_2 et notations standards)
    formule_clean = formule_str.replace('x_1', 'x1').replace('x_2', 'x2').replace('^', '**')
    
    try:
        expr = sp.sympify(formule_clean)
    except Exception as e:
        raise ValueError(f"Formule mathématique invalide : {e}")

    # Calcul analytique du gradient
    grad_expr = [sp.diff(expr, x1), sp.diff(expr, x2)]
    
    # Calcul analytique de la Hessienne
    hess_expr = [
        [sp.diff(grad_expr[0], x1), sp.diff(grad_expr[0], x2)],
        [sp.diff(grad_expr[1], x1), sp.diff(grad_expr[1], x2)]
    ]
    
    # Lambdification pour vitesse de calcul numpy
    f_lambda = sp.lambdify((x1, x2), expr, 'numpy')
    grad_lambda = [sp.lambdify((x1, x2), g, 'numpy') for g in grad_expr]
    hess_lambda = [[sp.lambdify((x1, x2), h_el, 'numpy') for h_el in row] for row in hess_expr]
    
    val = float(f_lambda(x1_val, x2_val))
    g_val = np.array([float(g(x1_val, x2_val)) for g in grad_lambda])
    h_val = np.array([[float(h_el(x1_val, x2_val)) for h_el in row] for row in hess_lambda])
    
    return val, g_val, h_val

def minimiser_fonction(formule_str, x0, max_iter=100, eps=1e-5, direction_method='gradient', step_method='optimal', fixed_alpha=0.1):
    """
    Algorithme global d'optimisation numérique 2D.
    Retourne la liste complète des itérations pour le tableau HTML et le cheminement.
    """
    iterations = []
    x_k = np.array([float(x0[0]), float(x0[1])])
    
    # BFGS state
    v_k = np.eye(2) # Inverse d'Hessienne initiale pour BFGS
    
    for k in range(max_iter + 1):
        # 1. Évaluation et stockage des dérivés
        try:
            val_k, grad_k, hess_k = evaluer_fonction_complete(formule_str, x_k[0], x_k[1])
        except Exception as e:
            iterations.append({
                'k': k, 'x': x_k.tolist(), 'f_x': None, 'grad': [0.0, 0.0],
                'grad_norm': 0.0, 'd': [0.0, 0.0], 'd_norm': 0.0,
                'alpha': 0.0, 'step_norm': 0.0, 'observation': f"Erreur : {e}"
            })
            break

        grad_norm = float(np.linalg.norm(grad_k))
        
        # Test de divergence géométrique ou dépassement de capacité
        if np.linalg.norm(x_k) > 1e5 or np.isnan(val_k) or np.isnan(grad_norm):
            iterations.append({
                'k': k, 'x': x_k.tolist(), 'f_x': val_k if not np.isnan(val_k) else 0.0,
                'grad': grad_k.tolist() if not np.any(np.isnan(grad_k)) else [0.0, 0.0],
                'grad_norm': grad_norm if not np.isnan(grad_norm) else 0.0,
                'd': [0.0, 0.0], 'd_norm': 0.0, 'alpha': 0.0, 'step_norm': 0.0,
                'observation': "Divergence géométrique (||x|| > 10^5)"
            })
            break

        # Test d'arrêt principal (tolérance epsilon)
        if grad_norm < eps:
            iterations.append({
                'k': k, 'x': x_k.tolist(), 'f_x': val_k, 'grad': grad_k.tolist(),
                'grad_norm': grad_norm, 'd': [0.0, 0.0], 'd_norm': 0.0,
                'alpha': 0.0, 'step_norm': 0.0, 'observation': "Convergence (gradient atteint)"
            })
            break

        if k == max_iter:
            iterations.append({
                'k': k, 'x': x_k.tolist(), 'f_x': val_k, 'grad': grad_k.tolist(),
                'grad_norm': grad_norm, 'd': [0.0, 0.0], 'd_norm': 0.0,
                'alpha': 0.0, 'step_norm': 0.0, 'observation': "Max itérations atteint"
            })
            break

        # 2. Détermination de la Direction d_k
        d_k = np.array([0.0, 0.0])
        obs_direction = ""
        
        if direction_method == 'gradient':
            d_k = -grad_k
            obs_direction = "Gradient descendant (Pente Forte)"
            
        elif direction_method == 'newton':
            # Newton direction: d_k = - inv(Hessienne) * grad_k
            # Sécurisation si Hessienne non inversible ou non définie positive
            det = hess_k[0, 0] * hess_k[1, 1] - hess_k[0, 1] * hess_k[1, 0]
            if abs(det) > 1e-12:
                try:
                    # Résolution de H_k * d_k = -grad_k
                    d_k = np.linalg.solve(hess_k, -grad_k)
                    # S'assurer que d_k est bien une direction de descente : d^T * grad < 0
                    if np.dot(d_k, grad_k) >= 0:
                        # Fallback gradient
                        d_k = -grad_k
                        obs_direction = "Newton (Non-convexe: repli sur gradient descendant)"
                    else:
                        obs_direction = "Newton (direction résolue analytiquement)"
                except np.linalg.LinAlgError:
                    d_k = -grad_k
                    obs_direction = "Newton (Hessienne singulière: repli sur gradient)"
            else:
                d_k = -grad_k
                obs_direction = "Newton (Hessienne singulière: repli sur gradient)"
                
        elif direction_method == 'bfgs':
            # BFGS: d_k = -V_k * grad_k
            if k == 0:
                v_k = np.eye(2) # réinitialisation explicite au point initial
            
            d_k = -np.dot(v_k, grad_k)
            # Vérification de la propriété de descente
            if np.dot(d_k, grad_k) >= 0:
                v_k = np.eye(2) # réinitialisation à l'identité si perdu
                d_k = -grad_k
                obs_direction = "BFGS (Alerte descente: réinitialisation à I)"
            else:
                obs_direction = f"BFGS (Approximation Hessienne)"

        d_norm = float(np.linalg.norm(d_k))
        if d_norm < 1e-15:
            iterations.append({
                'k': k, 'x': x_k.tolist(), 'f_x': val_k, 'grad': grad_k.tolist(),
                'grad_norm': grad_norm, 'd': d_k.tolist(), 'd_norm': d_norm,
                'alpha': 0.0, 'step_norm': 0.0, 'observation': "Direction nulle (stationnarité locale)"
            })
            break

        # 3. Calcul du Pas alpha_k
        alpha_k = 1.0
        obs_ls = ""
        
        if step_method == 'fixed':
            alpha_k = fixed_alpha
            obs_ls = f"Pas fixe ({fixed_alpha})"
            
        elif step_method == 'optimal':
            # Pas dynamique optimal pour formes quadratiques: alpha_k = - g^T * d / (d^T * H * d)
            num = -np.dot(grad_k, d_k)
            den = np.dot(d_k, np.dot(hess_k, d_k))
            if abs(den) > 1e-14 and (num / den) > 0:
                alpha_k = num / den
                obs_ls = "Pas optimal exact"
            else:
                alpha_k = 0.1 # pas de secours si dénum nul
                obs_ls = "Pas optimal estimé (secours non convexe)"
                
        elif step_method == 'armijo':
            # Règle d'Armijo : f(x + alpha*d) <= f(x) + c1 * alpha * grad^T * d
            c1 = 1e-4
            tau = 0.5
            alpha_k = 1.0
            ls_iter = 0
            g_t_d = np.dot(grad_k, d_k)
            lambda_f = sp.lambdify(sp.symbols('x1 x2'), sp.sympify(formule_str.replace('x_1', 'x1').replace('x_2', 'x2').replace('^', '**')), 'numpy')
            
            while ls_iter < 50:
                x_test = x_k + alpha_k * d_k
                try:
                    f_test = float(lambda_f(x_test[0], x_test[1]))
                except:
                    f_test = np.inf
                
                if f_test <= val_k + c1 * alpha_k * g_t_d:
                    break
                alpha_k *= tau
                ls_iter += 1
                
            obs_ls = f"Armijo (backtracks: {ls_iter})"
            
        elif step_method == 'wolfe':
            # Conditions de Wolfe (Faibles)
            # 1) f(x + alpha*d) <= f(x) + c1 * alpha * g^T * d (Armijo)
            # 2) grad f(x + alpha*d)^T * d >= c2 * g^T * d (Courbure)
            c1 = 1e-4
            c2 = 0.9
            g_t_d = np.dot(grad_k, d_k)
            
            alpha_low = 0.0
            alpha_high = np.inf
            alpha_k = 1.0
            ls_iter = 0
            
            while ls_iter < 50:
                x_test = x_k + alpha_k * d_k
                try:
                    f_test, grad_test, _ = evaluer_fonction_complete(formule_str, x_test[0], x_test[1])
                except Exception:
                    alpha_high = alpha_k
                    alpha_k = 0.5 * (alpha_low + alpha_high)
                    ls_iter += 1
                    continue
                
                if np.isnan(f_test) or f_test > val_k + c1 * alpha_k * g_t_d:
                    # Le pas est trop grand (ne satisfait pas Armijo)
                    alpha_high = alpha_k
                    alpha_k = 0.5 * (alpha_low + alpha_high)
                else:
                    # Armijo est ok, vérifions la courbure
                    g_test_t_d = np.dot(grad_test, d_k)
                    if g_test_t_d < c2 * g_t_d:
                        # Le pas est trop petit (slope trop négative)
                        alpha_low = alpha_k
                        if alpha_high == np.inf:
                            alpha_k = 2.0 * alpha_k
                        else:
                            alpha_k = 0.5 * (alpha_low + alpha_high)
                    else:
                        # Les deux conditions satisfaites
                        break
                ls_iter += 1
                
            obs_ls = f"Wolfe (iter LS: {ls_iter})"

        # Sécurisation du pas calculé
        if alpha_k <= 1e-15 or np.isnan(alpha_k):
            iterations.append({
                'k': k, 'x': x_k.tolist(), 'f_x': val_k, 'grad': grad_k.tolist(),
                'grad_norm': grad_norm, 'd': d_k.tolist(), 'd_norm': d_norm,
                'alpha': alpha_k if not np.isnan(alpha_k) else 0.0,
                'step_norm': 0.0, 'observation': "Pas trop petit (recherche linéaire bloquée)"
            })
            break

        # 4. Enregistrement de l'itération avant mise à jour
        x_next = x_k + alpha_k * d_k
        step_norm = float(np.linalg.norm(x_next - x_k))
        
        iterations.append({
            'k': k,
            'x': x_k.tolist(),
            'f_x': val_k,
            'grad': grad_k.tolist(),
            'grad_norm': grad_norm,
            'd': d_k.tolist(),
            'd_norm': d_norm,
            'alpha': alpha_k,
            'step_norm': step_norm,
            'observation': f"{obs_direction} | {obs_ls}"
        })

        # 5. Mise à jour de la matrice BFGS d'approx d'inverse de Hessienne
        if direction_method == 'bfgs':
            try:
                _, grad_next, _ = evaluer_fonction_complete(formule_str, x_next[0], x_next[1])
                s_k = x_next - x_k
                y_k = grad_next - grad_k
                
                ys = np.dot(y_k, s_k)
                if ys > 1e-10: # condition de courbure stricte pour préserver déf positive
                    rho_k = 1.0 / ys
                    I = np.eye(2)
                    # Formule d'update additive combinée : (I - rho*s*y^T) * V * (I - rho*y*s^T) + rho*s*s^T
                    term1 = I - rho_k * np.outer(s_k, y_k)
                    term2 = I - rho_k * np.outer(y_k, s_k)
                    v_k = np.dot(term1, np.dot(v_k, term2)) + rho_k * np.outer(s_k, s_k)
            except Exception:
                pass # Conserver l'ancienne matrice en cas d'erreur de calcul

        x_k = x_next
        
        # Test de convergence sur la stagnation des coordonnées
        if step_norm < 1e-15:
            iterations.append({
                'k': k + 1, 'x': x_k.tolist(), 'f_x': val_k, 'grad': grad_k.tolist(),
                'grad_norm': grad_norm, 'd': [0.0, 0.0], 'd_norm': 0.0,
                'alpha': 0.0, 'step_norm': 0.0, 'observation': "Convergence (Stationnarité)"
            })
            break
            
    return iterations
