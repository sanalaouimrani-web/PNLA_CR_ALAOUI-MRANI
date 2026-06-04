/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GraduationCap, Award } from 'lucide-react';

export default function Header() {
  return (
    <header className="bg-indigo-900 text-white px-6 py-4 flex flex-col md:flex-row justify-between items-center shadow-lg flex-shrink-0 border-b border-indigo-950">
      <div className="flex items-center space-x-3">
        <div>
          <h1 className="text-sm font-bold tracking-tight uppercase leading-none text-white">
            Université Moulay Ismaïl
          </h1>
          <p className="text-xs text-indigo-200 font-medium mt-1">
            Faculté des Sciences de Meknès • Master IARO
          </p>
        </div>
      </div>
      
      <div className="text-center md:text-right mt-3 md:mt-0 select-none">
        <h2 className="text-lg font-serif italic text-white leading-tight">
          Méthodes de Descente
        </h2>
        <p className="text-[10px] text-indigo-300 uppercase tracking-widest font-semibold mt-0.5">
          Modèles non-bornés & Convergence spectrale
        </p>
      </div>
    </header>
  );
}

