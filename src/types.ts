/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface Iteration {
  k: number;
  x: [number, number];       // x^k
  fVal: number;              // f(x^k)
  grad: [number, number];    // Grad f(x^k)
  gradNorm: number;          // ||Grad f(x^k)||
  d: [number, number];       // d_k (direction)
  dNorm: number;             // ||d_k||
  alpha: number;             // alpha_k (step length)
  stepNorm: number;          // ||x^{k+1} - x^k||
  observation: string;       // Observation text
}

export type DirectionMethod = 'gradient' | 'newton' | 'bfgs';
export type StepMethod = 'fixed' | 'optimal' | 'armijo' | 'wolfe';

export interface Config {
  functionId: string; // '1', '2', '3', '4', 'custom'
  customFormula: string;
  x0: [number, number];
  maxIter: number;
  epsilon: number;
  directionMethod: DirectionMethod;
  stepMethod: StepMethod;
  fixedAlpha: number;
}

export interface PreloadedFunction {
  id: string;
  title: string;
  formula: string;
  x0: [number, number];
  latex: string;
  description: string;
  tdReference: string;
}
