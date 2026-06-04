/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';
import { Iteration } from '../types';
import { evaluateFormula } from '../math_parser';
import { RefreshCw, Play, Volume2, Move, HelpCircle, Compass } from 'lucide-react';

interface ContourPlotProps {
  formula: string;
  iterations: Iteration[];
}

export default function ContourPlot({ formula, iterations }: ContourPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverCoord, setHoverCoord] = useState<[number, number] | null>(null);
  const [hoverVal, setHoverVal] = useState<number | null>(null);
  const [animationProgress, setAnimationProgress] = useState<number>(1); // 1 = fully drawn
  const [isAnimating, setIsAnimating] = useState<boolean>(false);

  // Derive bounding boxes with graceful padding
  let x1Min = -2;
  let x1Max = 2;
  let x2Min = -2;
  let x2Max = 2;

  if (iterations.length > 0) {
    const coords = iterations.map((it) => it.x);
    const x1Vals = coords.map((c) => c[0]);
    const x2Vals = coords.map((c) => c[1]);

    const x1Range = Math.max(...x1Vals) - Math.min(...x1Vals);
    const x2Range = Math.max(...x2Vals) - Math.min(...x2Vals);

    const pad1 = Math.max(1, x1Range * 0.35);
    const pad2 = Math.max(1, x2Range * 0.35);

    x1Min = Math.min(...x1Vals) - pad1;
    x1Max = Math.max(...x1Vals) + pad1;
    x2Min = Math.min(...x2Vals) - pad2;
    x2Max = Math.max(...x2Vals) + pad2;

    // Avoid zero-width bounding boxes
    if (x1Max - x1Min < 2) {
      x1Min -= 1;
      x1Max += 1;
    }
    if (x2Max - x2Min < 2) {
      x2Min -= 1;
      x2Max += 1;
    }
  }

  // Animation controller
  useEffect(() => {
    if (!isAnimating) return;
    let startTimestamp: number | null = null;
    const duration = 2000; // 2 seconds to trace the trajectory

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      setAnimationProgress(progress);
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        setIsAnimating(false);
      }
    };

    requestAnimationFrame(step);
  }, [isAnimating]);

  const triggerAnimation = () => {
    setAnimationProgress(0);
    setIsAnimating(true);
  };

  useEffect(() => {
    drawPlot();
  }, [formula, iterations, animationProgress, x1Min, x1Max, x2Min, x2Max]);

  const drawPlot = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // 1. Draw smooth continous heatmap background
    // To speed up evaluation, evaluate on an 80x80 grid and draw pixel rects
    const gridRows = 80;
    const gridCols = 80;
    const cellW = width / gridCols;
    const cellH = height / gridRows;

    const gridValues: number[][] = [];
    let fMin = Infinity;
    let fMax = -Infinity;

    for (let r = 0; r <= gridRows; r++) {
      gridValues[r] = [];
      const mathX2 = x2Max - (r / gridRows) * (x2Max - x2Min); // math standard y coordinate increases upwards
      for (let c = 0; c <= gridCols; c++) {
        const mathX1 = x1Min + (c / gridCols) * (x1Max - x1Min);
        let val = 0;
        try {
          val = evaluateFormula(formula, mathX1, mathX2).val;
          if (isNaN(val) || !isFinite(val)) val = 0;
        } catch {
          val = 0;
        }
        gridValues[r][c] = val;
        if (val < fMin) fMin = val;
        if (val > fMax) fMax = val;
      }
    }

    // Render cells onto background
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const val = gridValues[r][c];
        // Colormap normalization
        const norm = fMax === fMin ? 0 : (val - fMin) / (fMax - fMin);
        
        // Deep indigo/navy to pink/rose vector color ramp
        // r = 30 + 190 * norm
        // g = 40 + 80 * norm
        // b = 90 + 100 * (1 - norm)
        const red = Math.floor(15 + 175 * norm);
        const green = Math.floor(25 + 110 * norm);
        const blue = Math.floor(88 + 120 * (1 - norm));

        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // 2. Draw Vector Contour Isolines using MARCHING SQUARES
    // Draw 14 levels of isolines
    const levels = 14;
    ctx.lineWidth = 0.85;

    for (let l = 0; l < levels; l++) {
      const targetVal = fMin + ((l + 0.5) / levels) * (fMax - fMin);
      ctx.strokeStyle = `rgba(255, 255, 255, ${0.12 + 0.18 * (1 - l / levels)})`;

      for (let r = 0; r < gridRows; r++) {
        for (let c = 0; c < gridCols; c++) {
          // Corner values
          const v0 = gridValues[r][c];         // top-left
          const v1 = gridValues[r][c + 1];     // top-right
          const v2 = gridValues[r + 1][c + 1]; // bottom-right
          const v3 = gridValues[r + 1][c];     // bottom-left

          // Standard 4-bit config index
          let cellType = 0;
          if (v0 >= targetVal) cellType |= 8;
          if (v1 >= targetVal) cellType |= 4;
          if (v2 >= targetVal) cellType |= 2;
          if (v3 >= targetVal) cellType |= 1;

          // Corner viewport coordinates
          const xLeft = c * cellW;
          const xRight = (c + 1) * cellW;
          const yTop = r * cellH;
          const yBottom = (r + 1) * cellH;

          // Linear interpolation weights
          const lerpX = (va: number, vb: number) => {
            if (Math.abs(va - vb) < 1e-9) return 0.5;
            return (targetVal - va) / (vb - va);
          };

          const t_top = xLeft + lerpX(v0, v1) * cellW;
          const t_bottom = xLeft + lerpX(v3, v2) * cellW;
          const t_left = yTop + lerpX(v0, v3) * cellH;
          const t_right = yTop + lerpX(v1, v2) * cellH;

          ctx.beginPath();
          // Case-by-case marching squares segments
          switch (cellType) {
            case 1: case 14: // line bottom-left
              ctx.moveTo(xLeft, t_left); ctx.lineTo(t_bottom, yBottom);
              break;
            case 2: case 13: // line bottom-right
              ctx.moveTo(t_bottom, yBottom); ctx.lineTo(xRight, t_right);
              break;
            case 3: case 12: // horizontal divide
              ctx.moveTo(xLeft, t_left); ctx.lineTo(xRight, t_right);
              break;
            case 4: case 11: // line top-right
              ctx.moveTo(t_top, yTop); ctx.lineTo(xRight, t_right);
              break;
            case 5: // diagonal divide bottom-left & top-right
              ctx.moveTo(xLeft, t_left); ctx.lineTo(t_top, yTop);
              ctx.moveTo(t_bottom, yBottom); ctx.lineTo(xRight, t_right);
              break;
            case 6: case 9: // vertical divide
              ctx.moveTo(t_top, yTop); ctx.lineTo(t_bottom, yBottom);
              break;
            case 7: case 8: // line top-left
              ctx.moveTo(xLeft, t_left); ctx.lineTo(t_top, yTop);
              break;
            case 10: // diagonal divide top-left & bottom-right
              ctx.moveTo(xLeft, t_left); ctx.lineTo(t_bottom, yBottom);
              ctx.moveTo(t_top, yTop); ctx.lineTo(xRight, t_right);
              break;
          }
          ctx.stroke();
        }
      }
    }

    // Help Helper to convert math coords -> canvas pixel coordinates
    const toPixelX = (x1: number) => ((x1 - x1Min) / (x1Max - x1Min)) * width;
    const toPixelY = (x2: number) => height - ((x2 - x2Min) / (x2Max - x2Min)) * height;

    // Draw Axes (centered if origin lies in bounds)
    if (x1Min < 0 && x1Max > 0) {
      const axX = toPixelX(0);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(axX, 0); ctx.lineTo(axX, height);
      ctx.stroke();
    }
    if (x2Min < 0 && x2Max > 0) {
      const axY = toPixelY(0);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, axY); ctx.lineTo(width, axY);
      ctx.stroke();
    }

    // 3. Draw Iteration Trajectory & Arrows
    if (iterations.length > 0) {
      const maxPtsIdx = Math.floor((iterations.length - 1) * animationProgress) + 1;
      const ptsToDraw = iterations.slice(0, maxPtsIdx);

      ctx.strokeStyle = '#f43f5e'; // vivid crimson/pink trajectory line
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 2]);
      ctx.beginPath();
      ptsToDraw.forEach((pt, idx) => {
        const px = toPixelX(pt.x[0]);
        const py = toPixelY(pt.x[1]);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]); // reset

      // Draw arrows and points markers
      ptsToDraw.forEach((pt, idx) => {
        const px = toPixelX(pt.x[0]);
        const py = toPixelY(pt.x[1]);

        // Draw small nodes
        ctx.fillStyle = '#f43f5e';
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw connecting vector arrow indicating d_k * alpha_k
        if (idx < ptsToDraw.length - 1) {
          const nextPt = ptsToDraw[idx + 1];
          const npx = toPixelX(nextPt.x[0]);
          const npy = toPixelY(nextPt.x[1]);

          // Draw sharp arrow head
          const angle = Math.atan2(npy - py, npx - px);
          ctx.fillStyle = '#ffe4e6';
          ctx.save();
          ctx.translate(npx, npy);
          ctx.rotate(angle);
          ctx.beginPath();
          ctx.moveTo(0, 0);
          ctx.lineTo(-10, -5);
          ctx.lineTo(-7, 0);
          ctx.lineTo(-10, 5);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        }
      });

      // 4. Highlight Starting and Final point
      const sPx = toPixelX(iterations[0].x[0]);
      const sPy = toPixelY(iterations[0].x[1]);

      // Star gold for Initial point x0
      ctx.fillStyle = '#fbbf24';
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sPx, sPy, 8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // Inner text x0
      ctx.fillStyle = '#1e293b';
      ctx.font = 'bold 8px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('x0', sPx, sPx === 0 ? sPy : sPy);

      // Label x0
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px Inter';
      ctx.fillText('Initial x0', sPx, sPy - 14);

      // Stellar emerald for convergence target x*
      if (animationProgress === 1) {
        const finalPt = iterations[iterations.length - 1];
        const fPx = toPixelX(finalPt.x[0]);
        const fPy = toPixelY(finalPt.x[1]);

        ctx.fillStyle = '#10b981'; // emerald green
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(fPx, fPy, 10, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // Target dot
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(fPx, fPy, 3, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#22c55e';
        ctx.font = 'extrabold 11px Inter';
        ctx.fillText('Optimum x*', fPx, fPy - 16);
      }
    }
  };

  // Hover Coordinate Tracker inside canvas
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const mathX1 = x1Min + (px / canvas.width) * (x1Max - x1Min);
    const mathX2 = x2Max - (py / canvas.height) * (x2Max - x2Min);

    let val = 0;
    try {
      val = evaluateFormula(formula, mathX1, mathX2).val;
    } catch {
      val = NaN;
    }

    setHoverCoord([mathX1, mathX2]);
    setHoverVal(val);
  };

  const handleMouseLeave = () => {
    setHoverCoord(null);
    setHoverVal(null);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm flex flex-col relative overflow-hidden group">
      
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center gap-1.5 font-bold text-slate-500 text-xs tracking-wider uppercase">
          <Compass className="w-4 h-4 text-indigo-500" />
          <span>Graphique interactif</span>
        </div>

        <button
          onClick={triggerAnimation}
          disabled={isAnimating}
          className="flex items-center gap-1 text-[10px] font-bold text-slate-600 hover:text-indigo-600 transition-all bg-white border border-slate-250 hover:border-slate-400 px-2.5 py-1 rounded cursor-pointer"
        >
          <Play className="w-3 h-3 text-indigo-600 fill-indigo-600" />
          <span>Calculer l'animation</span>
        </button>
      </div>

      {/* Actual canvas */}
      <div className="relative aspect-square w-full rounded overflow-hidden border border-slate-200 bg-slate-50 shadow-inner flex justify-center items-center">
        <canvas
          ref={canvasRef}
          width={450}
          height={450}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="w-full h-full cursor-crosshair block"
        />

        {/* Live HUD Coordinate Overlay */}
        {hoverCoord && (
          <div className="absolute bottom-3 left-3 right-3 bg-white/95 backdrop-blur-sm border border-slate-200 p-2.5 rounded shadow-lg flex justify-between items-center text-[10px] font-mono text-slate-500 pointer-events-none transition-opacity">
            <div>
              <span className="text-slate-400 font-bold">x1:</span>{' '}
              <span className="text-slate-800 font-bold">{hoverCoord[0].toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate-400 font-bold">x2:</span>{' '}
              <span className="text-slate-800 font-bold">{hoverCoord[1].toFixed(4)}</span>
            </div>
            <div className="border-l border-slate-200 pl-3">
              <span className="text-indigo-600 font-bold">f(x):</span>{' '}
              <span className="text-indigo-700 font-bold">
                {isNaN(hoverVal ?? 0) ? 'Hors bornes' : (hoverVal ?? 0).toFixed(6)}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="text-[10px] text-slate-450 mt-2 text-center select-none leading-relaxed text-slate-400">
        Survolez le graphique pour explorer les amplitudes f(x1, x2). Les vecteurs bleus représentent <strong className="text-indigo-600 font-bold">d^k·α_k</strong>. Point de départ x0 en doré.
      </div>
    </div>
  );
}
