/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Copy, Check, FileCode, CheckSquare, Download } from 'lucide-react';

export default function DjangoWorkspace() {
  const [activeTab, setActiveTab] = useState<'opt' | 'view' | 'html'>('opt');
  const [copied, setCopied] = useState(false);

  const rawOptCode = `# -*- coding: utf-8 -*-
"""
Module d'Optimisation Numérique 2D pour le Master IARO.
Université Moulay Ismaïl, Faculté des Sciences de Meknès.
"""

import sympy as sp
import numpy as np

def evaluer_fonction_complete(formule_str, x1_val, x2_val):
    x1, x2 = sp.symbols('x1 x2')
    formule_clean = formule_str.replace('x_1', 'x1').replace('x_2', 'x2').replace('^', '**')
    expr = sp.sympify(formule_clean)

    grad_expr = [sp.diff(expr, x1), sp.diff(expr, x2)]
    hess_expr = [
        [sp.diff(grad_expr[0], x1), sp.diff(grad_expr[0], x2)],
        [sp.diff(grad_expr[1], x1), sp.diff(grad_expr[1], x2)]
    ]
    
    f_lambda = sp.lambdify((x1, x2), expr, 'numpy')
    grad_lambda = [sp.lambdify((x1, x2), g, 'numpy') for g in grad_expr]
    hess_lambda = [[sp.lambdify((x1, x2), h_el, 'numpy') for h_el in row] for row in hess_expr]
    
    val = float(f_lambda(x1_val, x2_val))
    g_val = np.array([float(g(x1_val, x2_val)) for g in grad_lambda])
    h_val = np.array([[float(h_el(x1_val, x2_val)) for h_el in row] for row in hess_lambda])
    
    return val, g_val, h_val

def minimiser_fonction(formule_str, x0, max_iter=100, eps=1e-5, direction_method='gradient', step_method='optimal', fixed_alpha=0.1):
    iterations = []
    x_k = np.array([float(x0[0]), float(x0[1])])
    v_k = np.eye(2) # Inverse d'Hessienne pour BFGS
    
    for k in range(max_iter + 1):
        try:
            val_k, grad_k, hess_k = evaluer_fonction_complete(formule_str, x_k[0], x_k[1])
        except Exception as e:
            iterations.append({
                'k': k, 'x': x_k.tolist(), 'f_x': None, 'grad': [0.0, 0.0],
                'grad_norm': 0.0, 'd': [0.0, 0.0], 'd_norm': 0.0, 'alpha': 0.0, 'step_norm': 0.0,
                'observation': f"Erreur : {e}"
            })
            break

        grad_norm = float(np.linalg.norm(grad_k))
        if np.linalg.norm(x_k) > 1e5 or np.isnan(val_k) or np.isnan(grad_norm):
            iterations.append({
                'k': k, 'x': x_k.tolist(), 'f_x': val_k, 'grad': grad_k.tolist(),
                'grad_norm': grad_norm, 'd': [0.0, 0.0], 'd_norm': 0.0, 'alpha': 0.0, 'step_norm': 0.0,
                'observation': "Divergence"
            })
            break

        if grad_norm < eps:
            iterations.append({
                'k': k, 'x': x_k.tolist(), 'f_x': val_k, 'grad': grad_k.tolist(),
                'grad_norm': grad_norm, 'd': [0.0, 0.0], 'd_norm': 0.0, 'alpha': 0.0, 'step_norm': 0.0,
                'observation': "Convergence (gradient atteint)"
            })
            break

        # Directions
        d_k = np.array([0.0, 0.0])
        if direction_method == 'gradient':
            d_k = -grad_k
        elif direction_method == 'newton':
            try:
                d_k = np.linalg.solve(hess_k, -grad_k)
                if np.dot(d_k, grad_k) >= 0:
                    d_k = -grad_k
            except:
                d_k = -grad_k
        elif direction_method == 'bfgs':
            d_k = -np.dot(v_k, grad_k)
            if np.dot(d_k, grad_k) >= 0:
                v_k = np.eye(2)
                d_k = -grad_k

        d_norm = float(np.linalg.norm(d_k))
        if d_norm < 1e-15:
            break

        # Pas
        alpha_k = 1.0
        if step_method == 'fixed':
            alpha_k = fixed_alpha
        elif step_method == 'optimal':
            den = np.dot(d_k, np.dot(hess_k, d_k))
            alpha_k = -np.dot(grad_k, d_k) / den if abs(den) > 1e-12 else 0.1
        elif step_method == 'armijo':
            # Règle d'Armijo
            c1, tau = 1e-4, 0.5
            while True:
                xt = x_k + alpha_k * d_k
                vt, _, _ = evaluer_fonction_complete(formule_str, xt[0], xt[1])
                if vt <= val_k + c1 * alpha_k * np.dot(grad_k, d_k):
                    break
                alpha_k *= tau

        x_next = x_k + alpha_k * d_k
        iterations.append({
            'k': k, 'x': x_k.tolist(), 'f_x': val_k, 'grad': grad_k.tolist(),
            'grad_norm': grad_norm, 'd': d_k.tolist(), 'd_norm': d_norm,
            'alpha': alpha_k, 'step_norm': float(np.linalg.norm(x_next - x_k)),
            'observation': f"{direction_method} | {step_method}"
        })

        if direction_method == 'bfgs':
            try:
                _, grad_next, _ = evaluer_fonction_complete(formule_str, x_next[0], x_next[1])
                s_k = x_next - x_k
                y_k = grad_next - grad_k
                rho_k = 1.0 / np.dot(y_k, s_k)
                I = np.eye(2)
                v_k = np.dot(I - rho_k*np.outer(s_k, y_k), np.dot(v_k, I - rho_k*np.outer(y_k, s_k))) + rho_k*np.outer(s_k, s_k)
            except:
                pass

        x_k = x_next
    return iterations`;

  const rawViewCode = `from django.shortcuts import render
from django.views import View
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from .optimization import minimiser_fonction

class OptimizationLabView(View):
    def get(self, request):
        return render(request, 'template.html', {'show_results': False})

    def post(self, request):
        formula = request.POST.get('formula')
        x0_x1 = float(request.POST.get('x0_x1', 0))
        x0_x2 = float(request.POST.get('x0_x2', 0))
        max_iter = int(request.POST.get('max_iter', 100))
        epsilon = float(request.POST.get('epsilon', 1e-5))
        direction_method = request.POST.get('direction_method', 'gradient')
        step_method = request.POST.get('step_method', 'optimal')
        fixed_alpha = float(request.POST.get('fixed_alpha', 0.1))

        iterations = minimiser_fonction(formula, (x0_x1, x0_x2), max_iter, epsilon, direction_method, step_method, fixed_alpha)
        chart_base64 = self.generer_graphique(formula, iterations)

        context = {
            'formula': formula, 'x0_x1': x0_x1, 'x0_x2': x0_x2,
            'max_iter': max_iter, 'epsilon': epsilon, 'direction_method': direction_method,
            'step_method': step_method, 'fixed_alpha': fixed_alpha,
            'iterations': iterations, 'chart_image': chart_base64, 'show_results': True
        }
        return render(request, 'template.html', context)

    def generer_graphique(self, formula_str, iterations):
        pts = np.array([pt['x'] for pt in iterations])
        x1_min, x1_max = pts[:, 0].min() - 1, pts[:, 0].max() + 1
        x2_min, x2_max = pts[:, 1].min() - 1, pts[:, 1].max() + 1
        
        # Maillage et tracé
        fig, ax = plt.subplots()
        ax.plot(pts[:, 0], pts[:, 1], color='red', linestyle='--', marker='o')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')`;

  const rawHtmlCode = `<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Laboratoire IARO</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50">
    <h1 class="text-xl font-bold p-5">Faculté des Sciences Meknès - Master IARO</h1>
    <!-- Convergence table -->
    <table class="w-full text-xs">
        <thead>
            <tr class="border-b-2 border-black">
                <th>k</th><th>x^k</th><th>f(x^k)</th><th>Grad f(x^k)</th><th>d^k</th><th>alpha^k</th><th>||Grad f(x^k)||</th><th>Observation</th>
            </tr>
        </thead>
        <tbody>
            {% for it in iterations %}
            <tr class="border-b">
                <td>{{ it.k }}</td>
                <td>({{ it.x.0|floatformat:4 }}, {{ it.x.1|floatformat:4 }})</td>
                <td>{{ it.f_x|floatformat:6 }}</td>
                <td>{{ it.grad_norm|floatformat:6 }}</td>
                <td>{{ it.observation }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>`;

  const getCode = () => {
    switch (activeTab) {
      case 'opt': return rawOptCode;
      case 'view': return rawViewCode;
      case 'html': return rawHtmlCode;
    }
  };

  const getFilename = () => {
    switch (activeTab) {
      case 'opt': return 'optimization.py';
      case 'view': return 'views.py';
      case 'html': return 'template.html';
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(getCode());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-100 pb-4 mb-4 gap-3">
        <div>
          <h3 className="text-xs font-bold uppercase text-slate-500 tracking-wider flex items-center gap-2">
            <FileCode className="w-4 h-4 text-indigo-600" />
            Espace Django & Python (TD3/TD4)
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Les fichiers sources requis générés de manière clé-en-main. Copiez-les directement dans votre projet Django d'évaluation.
          </p>
        </div>

        <div className="flex bg-slate-100 p-1 rounded border border-slate-200 self-stretch md:self-auto justify-around">
          <button
            onClick={() => setActiveTab('opt')}
            className={`px-3 py-1 rounded text-xs font-bold transition-all cursor-pointer ${
              activeTab === 'opt'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            optimization.py
          </button>
          <button
            onClick={() => setActiveTab('view')}
            className={`px-3 py-1 rounded text-xs font-bold transition-all cursor-pointer ${
              activeTab === 'view'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            views.py
          </button>
          <button
            onClick={() => setActiveTab('html')}
            className={`px-3 py-1 rounded text-xs font-bold transition-all cursor-pointer ${
              activeTab === 'html'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            template.html
          </button>
        </div>
      </div>

      {/* Code window */}
      <div className="relative bg-slate-950 rounded overflow-hidden border border-slate-800">
        <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800 bg-slate-900">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500/80"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/80"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-green-500/80"></span>
            <span className="text-[10px] font-mono font-bold text-slate-400 ml-2 select-none">
              {getFilename()}
            </span>
          </div>

          <button
            onClick={copyToClipboard}
            className="flex items-center gap-1.5 text-[10px] font-bold text-slate-300 hover:text-white transition-colors bg-slate-800 border border-slate-700 px-2.5 py-1 rounded cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 text-green-450 animate-bounce" />
                <span className="text-green-400 font-extrabold uppercase">Copié !</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                <span>Copier</span>
              </>
            )}
          </button>
        </div>

        <div className="p-4 overflow-x-auto text-[11px] font-mono leading-relaxed text-slate-300 bg-slate-950 max-h-[460px] scrollbar-thin select-all">
          <pre>{getCode()}</pre>
        </div>
      </div>

      <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between text-[11px] text-slate-500 font-medium gap-2">
        <span className="flex items-center gap-1.5">
          <CheckSquare className="w-3.5 h-3.5 text-emerald-600" />
          Validation de l'optimisation par SymPy analytique exacte et NumPy matriciel.
        </span>
        <span className="text-slate-400 font-semibold select-none">IARO FS Meknès</span>
      </div>
    </div>
  );
}
