/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { evaluateFormula, Dual2D } from './math_parser';
import { Iteration, DirectionMethod, StepMethod, Config } from './types';

// --- VECTOR OPERATIONS SIZED FOR 2D ---
const norm2D = (v: [number, number]): number => Math.sqrt(v[0] * v[0] + v[1] * v[1]);
const dot2D = (u: [number, number], v: [number, number]): number => u[0] * v[0] + u[1] * v[1];

// Helper to safely invert a 2x2 symmetric matrix H with regularization if singular or non-positive definite
// H = [[a, b], [b, d]]
// Return matrix inverse V
function invertHessian2D(H: [[number, number], [number, number]], betaDefault = 1e-4): [[number, number], [number, number]] | null {
  let a = H[0][0];
  let b = H[0][1];
  let d = H[1][1];

  // Levenberg-Marquardt style regularization to ensure positive-definiteness & invertibility
  // We compute eigenvalues to check for positive-definiteness
  // Eigenvalues of [[a, b], [b, d]]:
  // trace = a + d, det = ad - b^2
  // lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
  const trace = a + d;
  const det = a * d - b * b;
  const disc = trace * trace - 4 * det;
  const sqrtDisc = disc > 0 ? Math.sqrt(disc) : 0;
  const lambdaMin = (trace - sqrtDisc) / 2;

  // If minimum eigenvalue is negative or too small, matrix is not positive-definite
  if (lambdaMin < 1e-6) {
    // Add regularization parameter to diagonal
    const shift = Math.max(0, -lambdaMin) + betaDefault;
    a += shift;
    d += shift;
  }

  const cleanDet = a * d - b * b;
  if (Math.abs(cleanDet) < 1e-12) {
    return null; // completely singular
  }

  const invDet = 1.0 / cleanDet;
  return [
    [d * invDet, -b * invDet],
    [-b * invDet, a * invDet]
  ];
}

// Global optimization loop matching the COMBINATION of [Direction + Line Search]
export function runOptimization(formula: string, config: Config): Iteration[] {
  const iterations: Iteration[] = [];
  const { x0, maxIter, epsilon, directionMethod, stepMethod, fixedAlpha } = config;

  let xCurrent: [number, number] = [...x0];
  
  // BFGS State variable: inverse Hessian approximation. Starting as 2x2 identity matrix.
  let bfgsV: [[number, number], [number, number]] = [
    [1, 0],
    [0, 1]
  ];

  for (let k = 0; k <= maxIter; k++) {
    // 1. Evaluate function, gradient, and Hessian at current point
    let evalRes: Dual2D;
    try {
      evalRes = evaluateFormula(formula, xCurrent[0], xCurrent[1]);
    } catch (err: any) {
      iterations.push({
        k,
        x: xCurrent,
        fVal: NaN,
        grad: [NaN, NaN],
        gradNorm: NaN,
        d: [0, 0],
        dNorm: 0,
        alpha: 0,
        stepNorm: 0,
        observation: `Erreur d'évaluation: ${err.message}`,
      });
      break;
    }

    const { val: fVal, g: grad, h: Hessian } = evalRes;
    const gradNorm = norm2D(grad);

    // Check divergence first
    if (norm2D(xCurrent) > 1e5 || isNaN(fVal) || isNaN(gradNorm)) {
      iterations.push({
        k,
        x: xCurrent,
        fVal: isNaN(fVal) ? 0 : fVal,
        grad: grad,
        gradNorm: isNaN(gradNorm) ? 0 : gradNorm,
        d: [0, 0],
        dNorm: 0,
        alpha: 0,
        stepNorm: 0,
        observation: "Divergence géométrique (||x|| > 10^5)",
      });
      break;
    }

    // Check convergence criteria (||grad f|| < epsilon)
    if (gradNorm < epsilon) {
      iterations.push({
        k,
        x: xCurrent,
        fVal,
        grad,
        gradNorm,
        d: [0, 0],
        dNorm: 0,
        alpha: 0,
        stepNorm: 0,
        observation: "Convergence (gradient atteint)",
      });
      break;
    }

    if (k === maxIter) {
      iterations.push({
        k,
        x: xCurrent,
        fVal,
        grad,
        gradNorm,
        d: [0, 0],
        dNorm: 0,
        alpha: 0,
        stepNorm: 0,
        observation: "Max itérations atteint",
      });
      break;
    }

    // 2. Determine Search Direction d_k
    let d: [number, number] = [0, 0];
    let obsDirection = "";

    if (directionMethod === 'gradient') {
      // Steepest Descent: d = -g
      d = [-grad[0], -grad[1]];
      obsDirection = "Recherche selon le gradient descendant";
    } 
    else if (directionMethod === 'newton') {
      // Newton's Method: d = -H^{-1} * g
      const invH = invertHessian2D(Hessian);
      if (invH) {
        d = [
          -(invH[0][0] * grad[0] + invH[0][1] * grad[1]),
          -(invH[1][0] * grad[0] + invH[1][1] * grad[1])
        ];
        // For security, verify that it is actually a descent direction: d^T * grad < 0
        if (dot2D(d, grad) >= 0) {
          // Fallback to gradient if Hessian inversion gives ascent direction due to local non-convexity
          d = [-grad[0], -grad[1]];
          obsDirection = "Newton (Sensation de selle: repli sur gradient descendant)";
        } else {
          obsDirection = "Newton (direction résolue analytiquement)";
        }
      } else {
        // Fallback to gradient
        d = [-grad[0], -grad[1]];
        obsDirection = "Hessienne singulière (repli sur gradient)";
      }
    } 
    else if (directionMethod === 'bfgs') {
      // BFGS: d = -V_k * g
      if (k === 0) {
        // Initialize BFGS inverse Hessian as Identity
        bfgsV = [
          [1, 0],
          [0, 1]
        ];
      }

      d = [
        -(bfgsV[0][0] * grad[0] + bfgsV[0][1] * grad[1]),
        -(bfgsV[1][0] * grad[0] + bfgsV[1][1] * grad[1])
      ];

      // Verify descent direction
      if (dot2D(d, grad) >= 0) {
        // Reset BFGS matrix to identity
        bfgsV = [
          [1, 0],
          [0, 1]
        ];
        d = [
          -(bfgsV[0][0] * grad[0] + bfgsV[0][1] * grad[1]),
          -(bfgsV[1][0] * grad[0] + bfgsV[1][1] * grad[1])
        ];
        obsDirection = "BFGS (Alerte descente: réinitialisation à l'identité)";
      } else {
        obsDirection = `BFGS (Inverse H approx) [H00: ${bfgsV[0][0].toFixed(2)}]`;
      }
    }

    const dNorm = norm2D(d);
    if (dNorm < 1e-14) {
      iterations.push({
        k,
        x: xCurrent,
        fVal,
        grad,
        gradNorm,
        d,
        dNorm,
        alpha: 0,
        stepNorm: 0,
        observation: "Direction nulle obtenue (stationnarité locale)",
      });
      break;
    }

    // 3. Line Search to Determine Alpha_k
    let alpha = 1.0;
    let lsSucc = true;
    let obsLS = "";

    // Exact quadratic step: alpha = - (g^T d) / (d^T H d)
    const exactOptimalStep = (): number => {
      const numerator = -dot2D(grad, d);
      const denominator = d[0] * (Hessian[0][0] * d[0] + Hessian[0][1] * d[1]) +
                          d[1] * (Hessian[1][0] * d[0] + Hessian[1][1] * d[1]);
      if (Math.abs(denominator) < 1e-14 || numerator / denominator <= 0) {
        return 0.1; // defensive backup if not locally convex or singular
      }
      return numerator / denominator;
    };

    if (stepMethod === 'fixed') {
      alpha = fixedAlpha;
      obsLS = "Pas fixe";
    } 
    else if (stepMethod === 'optimal') {
      alpha = exactOptimalStep();
      obsLS = `Pas optimal exact`;
    } 
    else if (stepMethod === 'armijo') {
      // Armijo backtracking line search
      // f(x + alpha * d) <= f(x) + c1 * alpha * grad^T * d
      const c1 = 1e-4; // standard Armijo factor
      const tau = 0.5; // shrink ratio
      alpha = 1.0;
      let lsIter = 0;
      const gTd = dot2D(grad, d);

      while (lsIter < 50) {
        const testX: [number, number] = [xCurrent[0] + alpha * d[0], xCurrent[1] + alpha * d[1]];
        let fNew: number;
        try {
          fNew = evaluateFormula(formula, testX[0], testX[1]).val;
        } catch {
          fNew = Infinity; // safe backtrack
        }

        if (isNaN(fNew) || fNew > fVal + c1 * alpha * gTd) {
          alpha *= tau;
        } else {
          break; // condition satisfied
        }
        lsIter++;
      }

      if (lsIter >= 50) {
        lsSucc = false;
        obsLS = "Pas d'Armijo non convergé";
      } else {
        obsLS = `Armijo (backtracks: ${lsIter})`;
      }
    } 
    else if (stepMethod === 'wolfe') {
      // Weak Wolfe Line Search:
      // 1) Armijo: f(x + alpha*d) <= f(x) + c1 * alpha * g^T * d
      // 2) Curvature: grad f(x + alpha*d)^T * d >= c2 * g^T * d
      const c1 = 1e-4;
      const c2 = 0.9;
      const gTd = dot2D(grad, d);

      alpha = 1.0;
      let alphaLow = 0;
      let alphaHigh = Infinity;
      let lsIter = 0;

      while (lsIter < 50) {
        const testX: [number, number] = [xCurrent[0] + alpha * d[0], xCurrent[1] + alpha * d[1]];
        let nextRes: Dual2D;
        try {
          nextRes = evaluateFormula(formula, testX[0], testX[1]);
        } catch {
          // If evaluates fails, make it shrink
          alphaHigh = alpha;
          alpha = 0.5 * (alphaLow + alphaHigh);
          lsIter++;
          continue;
        }

        const fNew = nextRes.val;
        const gNew = nextRes.g;

        // Check sufficient decrease (Armijo)
        if (isNaN(fNew) || fNew > fVal + c1 * alpha * gTd) {
          alphaHigh = alpha;
          alpha = 0.5 * (alphaLow + alphaHigh);
        } else {
          // Sufficient decrease satisfied, now check curvature
          const gNewTd = dot2D(gNew, d);
          if (gNewTd < c2 * gTd) {
            // Slope is too steep negatives, we need a larger step
            alphaLow = alpha;
            if (alphaHigh === Infinity) {
              alpha = 2.0 * alpha; // double step size
            } else {
              alpha = 0.5 * (alphaLow + alphaHigh);
            }
          } else {
            // Both Weak Wolfe conditions met
            break;
          }
        }
        lsIter++;
      }

      if (lsIter >= 50) {
        // Fall back to Armijo if Wolfe is struggled
        alpha = 0.1;
        obsLS = "Pas de Wolfe (Échec: repli fuyant alpha=0.1)";
      } else {
        obsLS = `Wolfe (itérations LS: ${lsIter})`;
      }
    }

    // Safety checks on computed alpha
    if (!lsSucc || isNaN(alpha) || alpha <= 1e-15) {
      iterations.push({
        k,
        x: xCurrent,
        fVal,
        grad,
        gradNorm,
        d,
        dNorm,
        alpha: alpha || 0,
        stepNorm: 0,
        observation: "Recherche linéaire échouée / Oscillation",
      });
      break;
    }

    // 4. Update coordinates: x^{k+1} = x^k + alpha * d^k
    const xNext: [number, number] = [xCurrent[0] + alpha * d[0], xCurrent[1] + alpha * d[1]];
    const stepNorm = norm2D([xNext[0] - xCurrent[0], xNext[1] - xCurrent[1]]);

    // Record this complete step
    iterations.push({
      k,
      x: xCurrent,
      fVal,
      grad,
      gradNorm,
      d,
      dNorm,
      alpha,
      stepNorm,
      observation: `${obsDirection} -- ${obsLS}`,
    });

    // 5. Update BFGS Matrix V_{k+1} for the NEXT step using computed xNext
    if (directionMethod === 'bfgs') {
      try {
        const nextRes = evaluateFormula(formula, xNext[0], xNext[1]);
        const s_k: [number, number] = [xNext[0] - xCurrent[0], xNext[1] - xCurrent[1]];
        const y_k: [number, number] = [nextRes.g[0] - grad[0], nextRes.g[1] - grad[1]];

        const y_k_dot_s_k = dot2D(y_k, s_k);

        // Curvature condition for BFGS: y_k^T s_k > 0 to preserve positive definiteness
        if (y_k_dot_s_k > 1e-10) {
          const rho = 1.0 / y_k_dot_s_k;
          
          // Identity matrix
          const I: [[number, number], [number, number]] = [
            [1, 0],
            [0, 1]
          ];

          // Compute (I - rho * s_k * y_k^T)
          const M1: [[number, number], [number, number]] = [
            [I[0][0] - rho * s_k[0] * y_k[0], I[0][1] - rho * s_k[0] * y_k[1]],
            [I[1][0] - rho * s_k[1] * y_k[0], I[1][1] - rho * s_k[1] * y_k[1]]
          ];

          // Compute (I - rho * y_k * s_k^T)
          const M2: [[number, number], [number, number]] = [
            [I[0][0] - rho * y_k[0] * s_k[0], I[0][1] - rho * y_k[0] * s_k[1]],
            [I[1][0] - rho * y_k[1] * s_k[0], I[1][1] - rho * y_k[1] * s_k[1]]
          ];

          // Multiply M1 * bfgsV
          const T1: [[number, number], [number, number]] = [
            [
              M1[0][0] * bfgsV[0][0] + M1[0][1] * bfgsV[1][0],
              M1[0][0] * bfgsV[0][1] + M1[0][1] * bfgsV[1][1],
            ],
            [
              M1[1][0] * bfgsV[0][0] + M1[1][1] * bfgsV[1][0],
              M1[1][0] * bfgsV[0][1] + M1[1][1] * bfgsV[1][1],
            ]
          ];

          // Multiply T1 * M2
          const T2: [[number, number], [number, number]] = [
            [
              T1[0][0] * M2[0][0] + T1[0][1] * M2[1][0],
              T1[0][0] * M2[0][1] + T1[0][1] * M2[1][1],
            ],
            [
              T1[1][0] * M2[0][0] + T1[1][1] * M2[1][0],
              T1[1][0] * M2[0][1] + T1[1][1] * M2[1][1],
            ]
          ];

          // Add rho * s_k * s_k^T
          bfgsV = [
            [T2[0][0] + rho * s_k[0] * s_k[0], T2[0][1] + rho * s_k[0] * s_k[1]],
            [T2[1][0] + rho * s_k[1] * s_k[0], T2[1][1] + rho * s_k[1] * s_k[1]]
          ];
        }
      } catch {
        // Leave bfgsV unchanged if evaluation fails
      }
    }

    // Step to next coordinates
    xCurrent = xNext;

    // Check stationary step: ||x^{k+1} - x^k|| near machine zero to avoid looping on flat valleys
    if (stepNorm < 1e-15) {
      iterations.push({
        k: k + 1,
        x: xCurrent,
        fVal,
        grad,
        gradNorm,
        d: [0, 0],
        dNorm: 0,
        alpha: 0,
        stepNorm: 0,
        observation: "Convergence (Stationnarité: pas inférieur à 10^-15)",
      });
      break;
    }
  }

  return iterations;
}
