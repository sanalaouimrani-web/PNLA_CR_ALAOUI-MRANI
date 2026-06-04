/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

// --- DUAL NUMBER ARITHMETIC FOR AUTOMATIC DIFFERENTIATION (AD) ---
// Let u = (val, g, h)
// val: function value
// g: gradient vector [du_dx1, du_dx2]
// h: Hessian matrix element [[d2u_dx1x1, d2u_dx1x2], [d2u_dx1x2, d2u_dx2x2]]
export interface Dual2D {
  val: number;
  g: [number, number];
  h: [[number, number], [number, number]];
}

export function createConstant(c: number): Dual2D {
  return {
    val: c,
    g: [0, 0],
    h: [[0, 0], [0, 0]],
  };
}

export function createX1(x1: number): Dual2D {
  return {
    val: x1,
    g: [1, 0],
    h: [[0, 0], [0, 0]],
  };
}

export function createX2(x2: number): Dual2D {
  return {
    val: x2,
    g: [0, 1],
    h: [[0, 0], [0, 0]],
  };
}

export function add(u: Dual2D, v: Dual2D): Dual2D {
  return {
    val: u.val + v.val,
    g: [u.g[0] + v.g[0], u.g[1] + v.g[1]],
    h: [
      [u.h[0][0] + v.h[0][0], u.h[0][1] + v.h[0][1]],
      [u.h[1][0] + v.h[1][0], u.h[1][1] + v.h[1][1]],
    ],
  };
}

export function subtract(u: Dual2D, v: Dual2D): Dual2D {
  return {
    val: u.val - v.val,
    g: [u.g[0] - v.g[0], u.g[1] - v.g[1]],
    h: [
      [u.h[0][0] - v.h[0][0], u.h[0][1] - v.h[0][1]],
      [u.h[1][0] - v.h[1][0], u.h[1][1] - v.h[1][1]],
    ],
  };
}

export function multiply(u: Dual2D, v: Dual2D): Dual2D {
  const uv = u.val * v.val;
  // grad(u*v) = u * grad(v) + v * grad(u)
  const g0 = u.val * v.g[0] + v.val * u.g[0];
  const g1 = u.val * v.g[1] + v.val * u.g[1];

  // H(u*v) = u*H(v) + v*H(u) + grad(u)*grad(v)^T + grad(v)*grad(u)^T
  const h00 = u.val * v.h[0][0] + v.val * u.h[0][0] + 2 * u.g[0] * v.g[0];
  const h11 = u.val * v.h[1][1] + v.val * u.h[1][1] + 2 * u.g[1] * v.g[1];
  const h01 = u.val * v.h[0][1] + v.val * u.h[0][1] + u.g[0] * v.g[1] + u.g[1] * v.g[0];

  return {
    val: uv,
    g: [g0, g1],
    h: [
      [h00, h01],
      [h01, h11],
    ],
  };
}

export function reciprocal(u: Dual2D): Dual2D {
  if (u.val === 0) {
    throw new Error("Division par zéro détectée dans l'évaluation numérique.");
  }
  const s = 1.0 / u.val;
  const s2 = s * s;
  const s3 = s2 * s;

  const g0 = -s2 * u.g[0];
  const g1 = -s2 * u.g[1];

  const h00 = 2 * s3 * u.g[0] * u.g[0] - s2 * u.h[0][0];
  const h11 = 2 * s3 * u.g[1] * u.g[1] - s2 * u.h[1][1];
  const h01 = 2 * s3 * u.g[0] * u.g[1] - s2 * u.h[0][1];

  return {
    val: s,
    g: [g0, g1],
    h: [
      [h00, h01],
      [h01, h11],
    ],
  };
}

export function divide(u: Dual2D, v: Dual2D): Dual2D {
  return multiply(u, reciprocal(v));
}

export function power(u: Dual2D, p: number): Dual2D {
  if (p === 0) return createConstant(1);
  if (p === 1) return u;

  const val = Math.pow(u.val, p);

  // g = p * u^(p-1) * du/dx
  const coef1 = p * Math.pow(u.val, p - 1);
  // h_coef = p * (p-1) * u^(p-2)
  const coef2 = p * (p - 1) * Math.pow(u.val, p - 2);

  const g0 = coef1 * u.g[0];
  const g1 = coef1 * u.g[1];

  // H(u^p) = p * u^(p-1) * H(u) + p*(p-1) * u^(p-2) * grad(u)*grad(u)^T
  const h00 = coef1 * u.h[0][0] + (isNaN(coef2) ? 0 : coef2 * u.g[0] * u.g[0]);
  const h11 = coef1 * u.h[1][1] + (isNaN(coef2) ? 0 : coef2 * u.g[1] * u.g[1]);
  const h01 = coef1 * u.h[0][1] + (isNaN(coef2) ? 0 : coef2 * u.g[0] * u.g[1]);

  return {
    val: isNaN(val) ? 0 : val,
    g: [g0, g1],
    h: [
      [h00, h01],
      [h01, h11],
    ],
  };
}

export function exp(u: Dual2D): Dual2D {
  const val = Math.exp(u.val);
  const g0 = val * u.g[0];
  const g1 = val * u.g[1];

  const h00 = val * u.h[0][0] + val * u.g[0] * u.g[0];
  const h11 = val * u.h[1][1] + val * u.g[1] * u.g[1];
  const h01 = val * u.h[0][1] + val * u.g[0] * u.g[1];

  return {
    val,
    g: [g0, g1],
    h: [
      [h00, h01],
      [h01, h11],
    ],
  };
}

export function ln(u: Dual2D): Dual2D {
  if (u.val <= 0) {
    throw new Error("Logarithme d'une valeur négative ou nulle.");
  }
  const val = Math.log(u.val);
  const s = 1.0 / u.val;
  const s2 = s * s;

  const g0 = s * u.g[0];
  const g1 = s * u.g[1];

  const h00 = s * u.h[0][0] - s2 * u.g[0] * u.g[0];
  const h11 = s * u.h[1][1] - s2 * u.g[1] * u.g[1];
  const h01 = s * u.h[0][1] - s2 * u.g[0] * u.g[1];

  return {
    val,
    g: [g0, g1],
    h: [
      [h00, h01],
      [h01, h11],
    ],
  };
}

export function sin(u: Dual2D): Dual2D {
  const s = Math.sin(u.val);
  const c = Math.cos(u.val);

  const g0 = c * u.g[0];
  const g1 = c * u.g[1];

  // sin(u) Hessian
  const h00 = c * u.h[0][0] - s * u.g[0] * u.g[0];
  const h11 = c * u.h[1][1] - s * u.g[1] * u.g[1];
  const h01 = c * u.h[0][1] - s * u.g[0] * u.g[1];

  return {
    val: s,
    g: [g0, g1],
    h: [
      [h00, h01],
      [h01, h11],
    ],
  };
}

export function cos(u: Dual2D): Dual2D {
  const s = Math.sin(u.val);
  const c = Math.cos(u.val);

  const g0 = -s * u.g[0];
  const g1 = -s * u.g[1];

  const h00 = -s * u.h[0][0] - c * u.g[0] * u.g[0];
  const h11 = -s * u.h[1][1] - c * u.g[1] * u.g[1];
  const h01 = -s * u.h[0][1] - c * u.g[0] * u.g[1];

  return {
    val: c,
    g: [g0, g1],
    h: [
      [h00, h01],
      [h01, h11],
    ],
  };
}


// --- TOKENS FOR THE MATHEMATICAL LEXER ---
type TokenType = 'NUMBER' | 'VAR' | 'OP' | 'FUNC' | 'LPAREN' | 'RPAREN' | 'EOF';

interface Token {
  type: TokenType;
  value: string;
}

export function tokenize(str: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  // Let's normalize string first
  // Replace standard shorthand notations like 'x_1' or 'x_2' with 'x1' or 'x2'
  let parsedStr = str.replace(/x_1/g, 'x1').replace(/x_2/g, 'x2');
  // Lowercase representation of math
  parsedStr = parsedStr.toLowerCase();

  while (i < parsedStr.length) {
    const char = parsedStr[i];

    if (/\s/.test(char)) {
      i++;
      continue;
    }

    if (/[0-9.]/.test(char)) {
      let numStr = '';
      while (i < parsedStr.length && /[0-9.]/.test(parsedStr[i])) {
        numStr += parsedStr[i];
        i++;
      }
      tokens.push({ type: 'NUMBER', value: numStr });
      continue;
    }

    // Check variables: x1 or x2
    if (parsedStr.startsWith('x1', i)) {
      tokens.push({ type: 'VAR', value: 'x1' });
      i += 2;
      continue;
    }
    if (parsedStr.startsWith('x2', i)) {
      tokens.push({ type: 'VAR', value: 'x2' });
      i += 2;
      continue;
    }

    // Check math functions
    if (parsedStr.startsWith('sin', i)) {
      tokens.push({ type: 'FUNC', value: 'sin' });
      i += 3;
      continue;
    }
    if (parsedStr.startsWith('cos', i)) {
      tokens.push({ type: 'FUNC', value: 'cos' });
      i += 3;
      continue;
    }
    if (parsedStr.startsWith('exp', i)) {
      tokens.push({ type: 'FUNC', value: 'exp' });
      i += 3;
      continue;
    }
    if (parsedStr.startsWith('ln', i)) {
      tokens.push({ type: 'FUNC', value: 'ln' });
      i += 2;
      continue;
    }
    if (parsedStr.startsWith('log', i)) {
      tokens.push({ type: 'FUNC', value: 'ln' }); // treat log as ln by default
      i += 3;
      continue;
    }

    if (char === '(') {
      tokens.push({ type: 'LPAREN', value: '(' });
      i++;
      continue;
    }

    if (char === ')') {
      tokens.push({ type: 'RPAREN', value: ')' });
      i++;
      continue;
    }

    if (['+', '-', '*', '/', '^'].includes(char)) {
      tokens.push({ type: 'OP', value: char });
      i++;
      continue;
    }

    throw new Error(`Caractère non reconnu : '${char}' à la position ${i}`);
  }

  tokens.push({ type: 'EOF', value: '' });

  // Post-processing to insert implicit multiplications
  // (e.g. 2*x1 -> 2x1, or (x1+x2)(x1-x2) -> (x1+x2)*(x1-x2), or 2(x1+x2) -> 2*(x1+x2))
  const processedTokens: Token[] = [];
  for (let t = 0; t < tokens.length; t++) {
    const curr = tokens[t];
    processedTokens.push(curr);

    if (curr.type !== 'EOF' && t + 1 < tokens.length) {
      const next = tokens[t + 1];
      const insertMul =
        // NUMBER followed by VAR or FUNC or LPAREN
        (curr.type === 'NUMBER' && ['VAR', 'FUNC', 'LPAREN'].includes(next.type)) ||
        // VAR followed by VAR or FUNC or LPAREN
        (curr.type === 'VAR' && ['VAR', 'FUNC', 'LPAREN'].includes(next.type)) ||
        // RPAREN followed by VAR, FUNC, NUMBER, or LPAREN
        (curr.type === 'RPAREN' && ['VAR', 'FUNC', 'NUMBER', 'LPAREN'].includes(next.type));

      if (insertMul) {
        processedTokens.push({ type: 'OP', value: '*' });
      }
    }
  }

  return processedTokens;
}

// --- RECURSIVE DESCENT AST PARSER ---
export type ASTNode =
  | { type: 'NUMBER'; value: number }
  | { type: 'VAR'; value: 'x1' | 'x2' }
  | { type: 'UNARY_OP'; op: '-'; expr: ASTNode }
  | { type: 'BINARY_OP'; op: '+' | '-' | '*' | '/' | '^'; left: ASTNode; right: ASTNode }
  | { type: 'FUNC_CALL'; func: 'sin' | 'cos' | 'exp' | 'ln'; expr: ASTNode };

export function parse(tokens: Token[]): ASTNode {
  let index = 0;

  function peek(): Token {
    return tokens[index];
  }

  function consume(expectedType?: TokenType): Token {
    const t = tokens[index];
    if (expectedType && t.type !== expectedType) {
      throw new Error(`Erreur de syntaxe : Attendu '${expectedType}', obtenu '${t.type}' ('${t.value}')`);
    }
    index++;
    return t;
  }

  // Expression -> Term (('+' | '-') Term)*
  function parseExpression(): ASTNode {
    let node = parseTerm();
    while (peek().type === 'OP' && (peek().value === '+' || peek().value === '-')) {
      const opToken = consume();
      const right = parseTerm();
      node = {
        type: 'BINARY_OP',
        op: opToken.value as '+' | '-',
        left: node,
        right,
      };
    }
    return node;
  }

  // Term -> Factor (('*' | '/') Factor)*
  function parseTerm(): ASTNode {
    let node = parseFactor();
    while (peek().type === 'OP' && (peek().value === '*' || peek().value === '/')) {
      const opToken = consume();
      const right = parseFactor();
      node = {
        type: 'BINARY_OP',
        op: opToken.value as '*' | '/',
        left: node,
        right,
      };
    }
    return node;
  }

  // Factor -> Power ('^' Power)* [right-associative usually, let's keep it clean]
  function parseFactor(): ASTNode {
    let node = parsePrimary();
    if (peek().type === 'OP' && peek().value === '^') {
      consume();
      const right = parseFactor(); // Recursively call factor to support x^y^z right associativity
      node = {
        type: 'BINARY_OP',
        op: '^',
        left: node,
        right,
      };
    }
    return node;
  }

  // Primary -> NUMBER | VAR | FUNC '(' Expression ')' | '(' Expression ')' | '-' Primary
  function parsePrimary(): ASTNode {
    const t = peek();
    if (t.type === 'NUMBER') {
      consume();
      return { type: 'NUMBER', value: parseFloat(t.value) };
    }
    if (t.type === 'VAR') {
      consume();
      return { type: 'VAR', value: t.value as 'x1' | 'x2' };
    }
    if (t.type === 'FUNC') {
      consume();
      consume('LPAREN');
      const expr = parseExpression();
      consume('RPAREN');
      return { type: 'FUNC_CALL', func: t.value as 'sin' | 'cos' | 'exp' | 'ln', expr };
    }
    if (t.type === 'LPAREN') {
      consume();
      const expr = parseExpression();
      consume('RPAREN');
      return expr;
    }
    if (t.type === 'OP' && t.value === '-') {
      consume();
      const expr = parsePrimary();
      return { type: 'UNARY_OP', op: '-', expr };
    }
    if (t.type === 'OP' && t.value === '+') {
      consume();
      return parsePrimary();
    }

    throw new Error(`Erreur de syntaxe : Expression inattendue '${t.value || 'EOF'}'`);
  }

  const result = parseExpression();
  if (peek().type !== 'EOF') {
    throw new Error(`Erreur de syntaxe : Tokens non consommés à la fin de la formule.`);
  }
  return result;
}

// --- EVALUATOR WITH DUAL NUMBERS ---
export function evaluateAST(node: ASTNode, x1: number, x2: number): Dual2D {
  switch (node.type) {
    case 'NUMBER':
      return createConstant(node.value);

    case 'VAR':
      return node.value === 'x1' ? createX1(x1) : createX2(x2);

    case 'UNARY_OP': {
      const inside = evaluateAST(node.expr, x1, x2);
      return subtract(createConstant(0), inside);
    }

    case 'BINARY_OP': {
      const left = evaluateAST(node.left, x1, x2);
      const right = evaluateAST(node.right, x1, x2);
      switch (node.op) {
        case '+': return add(left, right);
        case '-': return subtract(left, right);
        case '*': return multiply(left, right);
        case '/': return divide(left, right);
        case '^': {
          // If right hand side is a constant numeric node, use pow directly for analytic derivatives
          if (node.right.type === 'NUMBER') {
            return power(left, node.right.value);
          } else {
            // General power u^v using exp(v * ln(u)), but typically powers inside TD are integers like 2
            return exp(multiply(right, ln(left)));
          }
        }
      }
      break;
    }

    case 'FUNC_CALL': {
      const inside = evaluateAST(node.expr, x1, x2);
      switch (node.func) {
        case 'sin': return sin(inside);
        case 'cos': return cos(inside);
        case 'exp': return exp(inside);
        case 'ln': return ln(inside);
      }
      break;
    }
  }
  throw new Error("Type de nœud AST non pris en charge.");
}

// Main high-level evaluator function
export function evaluateFormula(formula: string, x1: number, x2: number): Dual2D {
  try {
    const tokens = tokenize(formula);
    const ast = parse(tokens);
    return evaluateAST(ast, x1, x2);
  } catch (err: any) {
    throw new Error(`Formule mathématique invalide : ${err.message}`);
  }
}
