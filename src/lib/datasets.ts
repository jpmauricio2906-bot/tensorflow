// Synthetic datasets
export interface DataSet { xs: number[][]; ys: number[]; }

function addNoise([x,y]: [number, number], noise: number): [number, number] {
  if (noise <= 0) return [x,y];
  const nx = x + (Math.random() * 2 - 1) * noise;
  const ny = y + (Math.random() * 2 - 1) * noise;
  return [nx, ny];
}

export function genCircle(n=400, noise=0.05): DataSet {
  const xs: number[][] = []; const ys: number[] = [];
  for (let i=0;i<n;i++){
    const r = Math.random();
    const a = Math.random() * 2 * Math.PI;
    const x = Math.cos(a) * (0.2 + 0.8*r);
    const y = Math.sin(a) * (0.2 + 0.8*r);
    const label = (Math.hypot(x,y) < 0.5) ? 1 : 0;
    const [nx,ny] = addNoise([x,y], noise);
    xs.push([nx,ny]); ys.push(label);
  }
  return { xs, ys };
}

export function genXor(n=400, noise=0.05): DataSet {
  const xs: number[][] = []; const ys: number[] = [];
  for (let i=0;i<n;i++){
    const x = Math.random()*2-1;
    const y = Math.random()*2-1;
    const label = (x*y>0)?1:0;
    const [nx,ny] = addNoise([x,y], noise);
    xs.push([nx,ny]); ys.push(label);
  }
  return { xs, ys };
}

export function genGaussian(n=400, noise=0.05): DataSet {
  const xs: number[][] = []; const ys: number[] = [];
  const centers = [[-0.5,-0.5],[0.5,0.5]];
  for (let i=0;i<n;i++){
    const c = Math.random()<0.5?0:1;
    const x = centers[c][0] + randn()*0.3;
    const y = centers[c][1] + randn()*0.3;
    const [nx,ny] = addNoise([x,y], noise);
    xs.push([nx,ny]); ys.push(c);
  }
  return { xs, ys };
}

export function genSpiral(n=400, noise=0.05): DataSet {
  const xs: number[][] = []; const ys: number[] = [];
  const m = n/2;
  for(let i=0;i<m;i++){
    const r = i/m;
    const t = 4.5*r*Math.PI;
    const [x1,y1] = addNoise([ r*Math.cos(t), r*Math.sin(t) ], noise);
    const [x2,y2] = addNoise([ r*Math.cos(t+Math.PI), r*Math.sin(t+Math.PI) ], noise);
    xs.push([x1,y1]); ys.push(1);
    xs.push([x2,y2]); ys.push(0);
  }
  return { xs, ys };
}

export function genLinear(n=400, noise=0.05): DataSet {
  const xs: number[][] = []; const ys: number[] = [];
  const slope = 0.5; const intercept = 0;
  for (let i=0;i<n;i++){
    const x = Math.random()*2-1;
    const y = Math.random()*2-1;
    const label = y > slope*x + intercept ? 1 : 0;
    const [nx,ny] = addNoise([x,y], noise);
    xs.push([nx,ny]); ys.push(label);
  }
  return { xs, ys };
}

function randn() {
  let u=0, v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
  return Math.sqrt(-2.0*Math.log(u))*Math.cos(2.0*Math.PI*v);
}
