/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { 
  Play, 
  HelpCircle, 
  BookOpen, 
  Compass, 
  Database, 
  Grid, 
  ArrowRight,
  TrendingDown,
  Info
} from 'lucide-react';
import Header from './components/Header';
import TheoreticConclusions from './components/TheoreticConclusions';
import DjangoWorkspace from './components/DjangoWorkspace';
import ContourPlot from './components/ContourPlot';
import { runOptimization } from './optimization_engine';
import { evaluateFormula } from './math_parser';
import { Config, Iteration, PreloadedFunction } from './types';

const PRELOADED_FUNCTIONS: PreloadedFunction[] = [
  {
    id: '1',
    title: 'Fonction 1 (TD3 Ex 1&2)',
    formula: 'x1^2 + x1*x2 + x2^2',
    x0: [1.0, 1.0],
    latex: 'f(x_1, x_2) = x_1^2 + x_1 x_2 + x_2^2',
    description: 'Quadratique couplée. Permet d\'exposer le comportement oscillatoire caractéristique du gradient descendant linéaire.',
    tdReference: 'TD3 Exercices 1 & 2'
  },
  {
    id: '2',
    title: 'Fonction 2 (TD3 Ex 3&4)',
    formula: 'x1^2 + x2^2',
    x0: [2.0, 2.0],
    latex: 'f(x_1, x_2) = x_1^2 + x_2^2',
    description: 'Cuvette sphérique isotrope parfaite. Neutre et idéale, Newton ou gradient s\'y résolvent en un seul pas optimal.',
    tdReference: 'TD3 Exercices 3 & 4'
  },
  {
    id: '3',
    title: 'Fonction 3 (TD4 Ex 1&2)',
    formula: '(x1 - 1)^2 + 2*x2^2',
    x0: [0.0, 0.0],
    latex: 'f(x_1, x_2) = (x_1 - 1)^2 + 2 x_2^2',
    description: 'Quadratique décentrée avec anisotropie spectrale marquée d\'un facteur 2.',
    tdReference: 'TD4 Exercices 1 & 2'
  },
  {
    id: '4',
    title: 'Fonction 4 (TD4 Ex 3&4)',
    formula: '5*x1^2 + 6*x2^2 + 2*x1*x2 + 9*x1 + 5*x2',
    x0: [0.0, 0.0],
    latex: 'f(x_1, x_2) = 5 x_1^2 + 6 x_2^2 + 2 x_1 x_2 + 9 x_1 + 5 x_2',
    description: 'Forme quadratique arbitraire issue de devoirs d\'examens officiels de recherche opérationnelle.',
    tdReference: 'TD4 Exercices 3 & 4'
  }
];

export default function App() {
  const [config, setConfig] = useState<Config>({
    functionId: '1',
    customFormula: 'x1^2 + x1*x2 + x2^2',
    x0: [1.0, 1.0],
    maxIter: 50,
    epsilon: 0.00001,
    directionMethod: 'gradient',
    stepMethod: 'optimal',
    fixedAlpha: 0.1
  });

  const [formulaError, setFormulaError] = useState<string | null>(null);
  const [iterations, setIterations] = useState<Iteration[]>([]);
  const [isCustomFormula, setIsCustomFormula] = useState<boolean>(false);

  // Auto-run algorithm when configuration or formula changes
  useEffect(() => {
    runCalculation();
  }, [config, isCustomFormula]);

  const runCalculation = () => {
    setFormulaError(null);
    const activeFormula = isCustomFormula ? config.customFormula : getFormulaById(config.functionId);

    // Verify equation parsing first
    try {
      evaluateFormula(activeFormula, config.x0[0], config.x0[1]);
    } catch (err: any) {
      setFormulaError(err.message);
      setIterations([]);
      return;
    }

    const results = runOptimization(activeFormula, config);
    setIterations(results);
  };

  const getFormulaById = (id: string): string => {
    const fn = PRELOADED_FUNCTIONS.find(f => f.id === id);
    return fn ? fn.formula : 'x1^2 + x2^2';
  };

  const selectPreloadedFunction = (fn: PreloadedFunction) => {
    setIsCustomFormula(false);
    setConfig(prev => ({
      ...prev,
      functionId: fn.id,
      x0: [...fn.x0] as [number, number],
      customFormula: fn.formula
    }));
  };

  const handleCustomFormulaClick = () => {
    setIsCustomFormula(true);
    setConfig(prev => ({
      ...prev,
      functionId: 'custom'
    }));
  };

  const handleCustomFormulaChange = (val: string) => {
    setConfig(prev => ({
      ...prev,
      customFormula: val
    }));
  };

  const handleX0Change = (idx: number, valStr: string) => {
    const val = parseFloat(valStr);
    setConfig(prev => {
      const nextX0 = [...prev.x0] as [number, number];
      nextX0[idx] = isNaN(val) ? 0 : val;
      return {
        ...prev,
        x0: nextX0
      };
    });
  };

  const finalState = iterations[iterations.length - 1];

  return (
    <div className="bg-slate-100 text-slate-800 font-sans min-h-screen pb-16">
      
      {/* Banner */}
      <Header />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-10">

        {/* SECTION 1: PRELOADED FUNCTIONS DASHBOARD */}
        <section className="space-y-4">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 pb-2 border-b border-slate-200">
            <div>
              <h3 className="text-xs font-bold uppercase text-slate-500 tracking-wider">
                Bibliothèque de Fonctions (TD3/TD4)
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Sélectionnez une carte pour charger l'équation pré-calculée dans le laboratoire d'optimisation.
              </p>
            </div>
            <span className="text-[10px] bg-indigo-50 border border-indigo-150 text-indigo-700 px-2.5 py-1 rounded font-bold uppercase tracking-wider">
              Travaux Dirigés IARO
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {PRELOADED_FUNCTIONS.map((fn) => {
              const active = !isCustomFormula && config.functionId === fn.id;
              return (
                <div
                  id={`card-${fn.id}`}
                  key={fn.id}
                  onClick={() => selectPreloadedFunction(fn)}
                  className={`cursor-pointer rounded-lg p-4.5 transition-all duration-150 select-none ${
                    active
                      ? 'border-2 border-indigo-500 bg-indigo-50 shadow-sm'
                      : 'border border-slate-200 bg-slate-50 hover:border-indigo-300'
                  }`}
                >
                  <div className="flex justify-between text-[10px] font-bold mb-1 uppercase">
                    <span className={active ? 'text-indigo-500' : 'text-slate-400'}>{fn.tdReference}</span>
                    <span className={active ? 'text-indigo-400/80' : 'text-slate-400'}>x0 = ({fn.x0[0]}, {fn.x0[1]})</span>
                  </div>
                  
                  <code className="text-xs font-mono font-bold block bg-white/80 border border-slate-200/50 rounded p-2 my-2 text-slate-900 select-all">
                    {fn.formula}
                  </code>
                  
                  <p className="text-[11px] text-slate-500 leading-normal min-h-[32px] line-clamp-2">
                    {fn.description}
                  </p>
                </div>
              );
            })}
          </div>
        </section>

        {/* SECTION 2: CUSTOM FORMULA ENTRY */}
        <section className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div>
              <h3 className="text-xs font-bold uppercase text-slate-500 tracking-wider">
                Saisie d'une formulation personnalisée f(x1, x2)
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Définissez votre propre forme mathématique à étudier.
              </p>
            </div>
            <button
              onClick={handleCustomFormulaClick}
              className={`text-xs font-bold px-3.5 py-1.5 rounded border transition-all ${
                isCustomFormula
                  ? 'bg-indigo-600 border-indigo-600 text-white shadow-sm'
                  : 'bg-indigo-50 hover:bg-indigo-100 text-indigo-600 border-indigo-200'
              }`}
            >
              + Saisir une fonction personnalisée
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-1.5">
              <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider">Expression de la fonction</label>
              <input
                type="text"
                value={isCustomFormula ? config.customFormula : getFormulaById(config.functionId)}
                readOnly={!isCustomFormula}
                onChange={(e) => handleCustomFormulaChange(e.target.value)}
                placeholder="Exemple: 5*x1^2 + 6*x2^2 + 2*x1*x2 + 9*x1"
                className={`w-full font-mono font-bold text-xs p-2.5 rounded border outline-none transition-all ${
                  isCustomFormula 
                    ? 'bg-white text-slate-900 border-indigo-300 focus:ring-2 focus:ring-indigo-500' 
                    : 'bg-slate-100 text-slate-500 border-slate-200 cursor-not-allowed select-none'
                }`}
              />
              <p className="text-[11px] text-slate-400 mt-1">
                Variables gérées: <code className="bg-slate-100 text-indigo-600 px-1 py-0.5 rounded font-mono">x1</code> et <code className="bg-slate-100 text-indigo-600 px-1 py-0.5 rounded font-mono">x2</code>. Opérateurs: <code className="font-mono text-slate-600 font-bold">+ - * / ^</code>. Supporte la dérivation automatique.
              </p>
            </div>

            {/* Custom Coordinates Initial */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider">Point initial x0_1</label>
                <input
                  type="number"
                  step="any"
                  value={config.x0[0]}
                  onChange={(e) => handleX0Change(0, e.target.value)}
                  className="w-full text-xs font-bold p-2.5 rounded border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div className="space-y-1.5">
                <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider">Point initial x0_2</label>
                <input
                  type="number"
                  step="any"
                  value={config.x0[1]}
                  onChange={(e) => handleX0Change(1, e.target.value)}
                  className="w-full text-xs font-bold p-2.5 rounded border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
          </div>

          {formulaError && (
            <div className="bg-red-50 text-red-700 text-xs mt-3 p-3 rounded border border-red-200 font-medium font-mono">
              Erreur lexicale ou de syntaxe : {formulaError}
            </div>
          )}
        </section>

        {/* SECTION 3: PARAMETERS AND MODEL ALGORITHMS */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Box 1: Stopping boundaries */}
          <div className="bg-white rounded-lg border border-slate-200 p-5 shadow-sm space-y-4">
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-100 pb-2">
              Limites de calculs & Critères
            </h4>
            <div className="space-y-3">
              <div className="space-y-1">
                <label className="block text-[9px] font-bold text-slate-500 uppercase">Max Iterations (Entier)</label>
                <input
                  type="number"
                  min="1"
                  max="1000"
                  value={config.maxIter}
                  onChange={(e) => setConfig(prev => ({ ...prev, maxIter: parseInt(e.target.value) || 50 }))}
                  className="w-full bg-white border border-slate-300 rounded p-2 text-xs font-semibold focus:ring-2 focus:ring-indigo-500 outline-none text-slate-800"
                />
                <p className="text-[10px] text-slate-400 mt-1">Nombre d'itérations butoir pour couper la boucle.</p>
              </div>

              <div className="space-y-1">
                <label className="block text-[9px] font-bold text-slate-500 uppercase">Tolerance epsilon (ε)</label>
                <input
                  type="number"
                  step="any"
                  value={config.epsilon}
                  onChange={(e) => setConfig(prev => ({ ...prev, epsilon: parseFloat(e.target.value) || 0.0001 }))}
                  className="w-full bg-white border border-slate-300 rounded p-2 text-xs font-semibold focus:ring-2 focus:ring-indigo-500 outline-none text-slate-800"
                />
                <p className="text-[10px] text-slate-400 mt-1">Seuil minimal d'arrêt sur la norme euclidienne du gradient.</p>
              </div>
            </div>
          </div>

          {/* Box 2: Desent direction choice */}
          <div className="bg-white rounded-lg border border-slate-200 p-5 shadow-sm space-y-4">
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-100 pb-2">
              Algorithme de Direction (d^k)
            </h4>
            <div className="space-y-3">
              <div className="space-y-1">
                <label className="block text-[9px] font-bold text-slate-500 uppercase">Méthode de direction</label>
                <select
                  value={config.directionMethod}
                  onChange={(e) => setConfig(prev => ({ ...prev, directionMethod: e.target.value as any }))}
                  className="w-full bg-white border border-slate-300 rounded p-2 text-xs font-bold focus:ring-2 focus:ring-indigo-500 outline-none text-slate-700"
                >
                  <option value="gradient">Gradient Descendant (Pente)</option>
                  <option value="newton">Méthode de Newton</option>
                  <option value="bfgs">BFGS (Quasi-Newton)</option>
                </select>
                <p className="text-[10px] text-slate-400 mt-1.5 leading-relaxed">
                  Gradient utilise <code className="font-mono bg-slate-100 text-slate-600 px-1 py-0.5 rounded">-g</code>. Newton résout analytiquement via la Hessienne locale inversée. BFGS l'approxime de façon additive.
                </p>
              </div>
            </div>
          </div>

          {/* Box 3: Line search step sizing */}
          <div className="bg-white rounded-lg border border-slate-200 p-5 shadow-sm space-y-4">
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-100 pb-2">
              Recherche Linéaire (Pas α)
            </h4>
            <div className="space-y-3">
              <div className="space-y-1">
                <label className="block text-[9px] font-bold text-slate-500 uppercase">Choix du pas</label>
                <select
                  value={config.stepMethod}
                  onChange={(e) => setConfig(prev => ({ ...prev, stepMethod: e.target.value as any }))}
                  className="w-full bg-white border border-slate-300 rounded p-2 text-xs font-bold focus:ring-2 focus:ring-indigo-500 outline-none text-slate-700"
                >
                  <option value="fixed">Pas Fixe</option>
                  <option value="optimal">Pas Dynamique Optimal</option>
                  <option value="armijo">Règle d'Armijo</option>
                  <option value="wolfe">Conditions de Wolfe</option>
                </select>
              </div>

              {config.stepMethod === 'fixed' && (
                <div className="space-y-1 animate-fadeIn">
                  <label className="block text-[9px] font-bold text-slate-500 uppercase">Pas initial α</label>
                  <input
                    type="number"
                    step="any"
                    value={config.fixedAlpha}
                    onChange={(e) => setConfig(prev => ({ ...prev, fixedAlpha: parseFloat(e.target.value) || 0.1 }))}
                    className="w-full bg-white border border-slate-300 rounded p-2 text-xs font-semibold focus:ring-2 focus:ring-indigo-500 outline-none text-slate-800"
                  />
                  <p className="text-[10px] text-slate-400 mt-1">
                    Un pas excessif engendre une instabilité spectrale immédiate.
                  </p>
                </div>
              )}
            </div>
          </div>

        </section>

        {/* SECTION 4: THEORETIC ROADMAP NOTES */}
        <TheoreticConclusions />

        {/* SECTION 5: LIVE SIMULATION DISPLAY (CANVAS CONTOURS + HUD) */}
        {!formulaError && iterations.length > 0 && (
          <section className="space-y-6">
            <div className="border-t border-slate-200 pt-8">
              <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Résultats de la Simulation Numérique</h2>
              <p className="text-xs text-slate-500">Visualisation 2D vectorielle et convergence calculée de votre modèle.</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              
              {/* Left Contour Plot Column */}
              <div className="lg:col-span-5">
                <ContourPlot 
                  formula={isCustomFormula ? config.customFormula : getFormulaById(config.functionId)} 
                  iterations={iterations} 
                />
              </div>

              {/* Right Convergent Metrics Column */}
              <div className="lg:col-span-7 bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-6">
                <div className="flex items-center gap-2 font-bold text-slate-800 text-sm tracking-wider uppercase">
                  <TrendingDown className="w-4 h-4 text-indigo-600" />
                  <span>État de Convergence global</span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="bg-slate-50 p-4 rounded-xl text-center border border-slate-100">
                    <span className="block text-[9px] font-bold text-slate-400 uppercase tracking-widest leading-none">Itérations</span>
                    <span className="text-2xl font-extrabold text-indigo-600 mt-1.5 block leading-none">
                      {Math.max(0, iterations.length - 1)}
                    </span>
                  </div>
                  <div className="bg-slate-50 p-4 rounded-xl text-center border border-slate-100">
                    <span className="block text-[9px] font-bold text-slate-400 uppercase tracking-widest leading-none">Minimun f(x*)</span>
                    <span className="text-base font-extrabold text-slate-950 mt-2 block font-mono leading-none">
                      {finalState?.fVal.toFixed(6) ?? 'NaN'}
                    </span>
                  </div>
                  <div className="bg-slate-50 p-4 rounded-xl text-center border border-slate-100">
                    <span className="block text-[9px] font-bold text-slate-400 uppercase tracking-widest leading-none">Optimum x*</span>
                    <span className="text-xs font-bold text-emerald-700 mt-2.5 block font-mono leading-none">
                      ({finalState?.x[0].toFixed(3) ?? '0'}, {finalState?.x[1].toFixed(3) ?? '0'})
                    </span>
                  </div>
                  <div className="bg-slate-50 p-4 rounded-xl text-center border border-slate-100">
                    <span className="block text-[9px] font-bold text-slate-400 uppercase tracking-widest leading-none">Grad Final</span>
                    <span className="text-xs font-bold text-slate-950 mt-2.5 block font-mono leading-none">
                      {finalState?.gradNorm.toFixed(6) ?? '0'}
                    </span>
                  </div>
                </div>

                <div className="border-t border-slate-100 pt-5 text-xs text-slate-600 leading-relaxed bg-indigo-50/40 p-4.5 rounded-xl border border-indigo-100/60 font-medium">
                  <span className="font-extrabold text-indigo-950 block mb-1 uppercase tracking-widest text-[10px]">Observation Générale de l'Optimisation :</span>
                  <strong>Conclusion de l'étape :</strong> {finalState?.observation}.
                  {finalState?.observation.includes('Convergence') ? (
                    <p className="mt-1 text-slate-500">
                      L'algorithme de descente a convergé vers un point stationnaire KKT. La tolérance programmée de <code className="bg-white px-1 py-0.5 rounded border text-indigo-600 font-mono">{config.epsilon}</code> est vérifiée.
                    </p>
                  ) : finalState?.observation.includes('Divergence') ? (
                    <p className="mt-1 text-red-700 font-semibold bg-red-50 p-2 rounded border border-red-100/60">
                      Alerte Divergence : La norme a divergé, cela se produit généralement si votre pas fixe est au-delà du seuil spectral de stabilité.
                    </p>
                  ) : (
                    <p className="mt-1 text-slate-500">
                      L'algorithme a stoppé car les ressources de calculs ont été atteintes sans pouvoir valider le critère sur la norme du gradient de f(x).
                    </p>
                  )}
                </div>
              </div>

            </div>

            {/* HIGH FIDELITY TABLE CONFORM TO USER REQUIREMENTS */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden p-6">
              <div className="flex justify-between items-center pb-4 mb-4 border-b border-slate-100">
                <h3 className="text-sm font-extrabold text-slate-900 uppercase tracking-wider">
                  Tableau de Simulation (Norme Académique Booktabs)
                </h3>
                <span className="text-[10px] bg-slate-100 text-slate-500 font-bold px-2.5 py-1 rounded-full uppercase tracking-widest">
                  Calculs Exacts (AD)
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-[11px] font-mono whitespace-nowrap select-all table-auto">
                  <thead>
                    <tr className="border-t-2 border-b-1.5 border-slate-900 bg-slate-50/50 text-slate-900 text-xs font-semibold">
                      <th className="py-3 px-3 first:pl-4">k</th>
                      <th className="py-3 px-3">x^k</th>
                      <th className="py-3 px-3">f(x^k)</th>
                      <th className="py-3 px-3">Grad f(x^k)</th>
                      <th className="py-3 px-3">d^k</th>
                      <th className="py-3 px-3">alpha^k</th>
                      <th className="py-3 px-3">||d^k||</th>
                      <th className="py-3 px-3">||x^(k+1) - x^k||</th>
                      <th className="py-3 px-3">||Grad f(x^k)||</th>
                      <th className="py-3 px-3 font-sans font-bold text-slate-900/90 pr-4 last:pr-4">Observation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {iterations.map((it) => (
                      <tr key={it.k} className="hover:bg-slate-50/60 transition-colors">
                        <td className="py-3 px-3 first:pl-4 font-bold text-slate-950">{it.k}</td>
                        <td className="py-3 px-3 text-indigo-900">
                          ({it.x[0].toFixed(4)}, {it.x[1].toFixed(4)})
                        </td>
                        <td className="py-3 px-3 font-bold text-slate-900">
                          {isNaN(it.fVal) ? 'NaN' : it.fVal.toFixed(6)}
                        </td>
                        <td className="py-3 px-3 text-slate-500">
                          ({it.grad[0].toFixed(4)}, {it.grad[1].toFixed(4)})
                        </td>
                        <td className="py-3 px-3 text-slate-500">
                          ({it.d[0].toFixed(4)}, {it.d[1].toFixed(4)})
                        </td>
                        <td className="py-3 px-3 font-bold text-emerald-800">
                          {it.alpha.toFixed(6)}
                        </td>
                        <td className="py-3 px-3 text-slate-500">{it.dNorm.toFixed(6)}</td>
                        <td className="py-3 px-3 text-slate-500">{it.stepNorm.toFixed(6)}</td>
                        <td className="py-3 px-3 font-bold text-slate-950">{it.gradNorm.toFixed(6)}</td>
                        <td className="py-3 px-3 font-sans text-slate-700 font-semibold pr-4 last:pr-4">
                          {it.observation}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="border-b-2 border-slate-900">
                      <td colSpan={10} className="py-1"></td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </div>

          </section>
        )}

        {/* SECTION 6: READY-TO-GET DJANGO FOLDER SOURCE CODE FOR TASKS */}
        <DjangoWorkspace />

      </main>

      {/* Footer */}
      <footer className="mt-20 text-center text-xs text-slate-400 font-semibold border-t border-slate-205/60 pt-8 max-w-7xl mx-auto">
        <p>© 2026 Université Moulay Ismaïl — Faculté des Sciences de Meknès — Master IARO.</p>
      </footer>

    </div>
  );
}
