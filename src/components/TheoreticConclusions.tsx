/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Lightbulb, Info, AlertTriangle, Cpu } from 'lucide-react';

export default function TheoreticConclusions() {
  return (
    <div className="bg-white text-slate-800 rounded-lg border border-slate-200 p-6 shadow-sm">
      <div className="flex items-center gap-2 mb-4 border-b border-slate-100 pb-3">
        <Lightbulb className="w-5 h-5 text-amber-500" />
        <h3 className="text-xs font-bold uppercase text-slate-500 tracking-wider">
          Conclusions Théoriques — Recherche Opérationnelle & Convergence
        </h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 text-[11px] leading-relaxed">
        {/* Card 1 */}
        <div className="bg-slate-50 p-4.5 rounded-lg border border-slate-200 hover:border-indigo-300 transition-colors">
          <div className="flex items-center gap-1.5 font-bold mb-2 text-indigo-700">
            <Info className="w-4 h-4 text-indigo-500" />
            <span>Spectre, Conditionnement et Zigzag</span>
          </div>
          <p className="text-slate-600 font-medium">
            Le taux de convergence locale du Gradient Descendant est intrinsèquement lié au{" "}
            <strong>conditionnement spectral</strong> de la matrice Hessienne {"\\( H_f(x) \\)"}. 
            Si le ratio {"\\( \\kappa = \\frac{\\lambda_{max}}{\\lambda_{min}} \\)"} est largement supérieur à 1, 
            les lignes de niveau forment un ravin étiré (anisotropie). Le vecteur gradient pointera 
            presque orthogonalement au chemin direct, provoquant des <strong>oscillations en zigzag</strong> inefficaces.
          </p>
        </div>

        {/* Card 2 */}
        <div className="bg-indigo-50/50 p-4.5 rounded-lg border border-indigo-150/60 hover:border-indigo-300 transition-colors">
          <div className="flex items-center gap-1.5 font-bold mb-2 text-indigo-800">
            <Cpu className="w-4 h-4 text-indigo-600" />
            <span>Métrique Newton & Quadratique</span>
          </div>
          <p className="text-slate-700 font-medium font-sans">
            La méthode de Newton contourne le problème de conditionnement en effectuant une rotation 
            de la direction par l'inverse de la Hessienne analytique : {"\\( d^k = - [H_f(x^k)]^{-1} \\nabla f(x^k) \\)"}. 
            Cela équivaut à déformer localement la topographie de l'espace pour restaurer une symétrie circulaire (isomorphe). 
            Sur une forme quadratique stricte avec <strong>Pas Optimal Exact</strong>, le modèle converge 
            en <strong>une unique itération</strong> sans aucun zigzag.
          </p>
        </div>

        {/* Card 3 */}
        <div className="bg-amber-50 border border-amber-200 rounded p-4.5 hover:border-amber-300 transition-colors">
          <div className="flex items-center gap-1.5 font-bold mb-2 text-amber-800">
            <AlertTriangle className="w-4 h-4 text-amber-600" />
            <span>Recherche Linéaire & Divergence</span>
          </div>
          <p className="text-amber-900 font-medium">
            Si le pas fixe {"\\( \\alpha \\)"} est choisi au-dessus du rayon de convergence 
            théorique {"\\( \\alpha > 2 / \\lambda_{max} \\)"}, les itérations divergent instantanément 
            vers l'infini à cause de phénomènes d'amplification spectrale. L'intégration de la{" "}
            <strong>Règle d'Armijo</strong> ou des <strong>Conditions de Wolfe</strong> assure une décroissance 
            suffisante et empêche la recherche de stagner ou de diverger en adaptant dynamiquement le pas.
          </p>
        </div>
      </div>
    </div>
  );
}

