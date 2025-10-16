export interface DataSet { xs: number[][]; ys: number[]; }

function addNoise([x,y]: [number, number], n: number): [number, number] {
  if (n <= 0) return [x,y];
  return [x + (Math.random()*2-1)*n, y + (Math.random()*2-1)*n];
}

export function genCircle(m=400, noise=0.05): DataSet {
  const xs: number[][] = [], ys: number[] = [];
  for (let i=0;i<m;i++){
    const r = Math.random(), a = Math.random()*2*Math.PI;
    const x = Math.cos(a)*(0.2+0.8*r), y = Math.sin(a)*(0.2+0.8*r);
    const lab = Math.hypot(x,y) < 0.5 ? 1 : 0;
    const [nx,ny] = addNoise([x,y], noise);
    xs.push([nx,ny]); ys.push(lab);
  }
  return { xs, ys };
}

export function genXor(m=400, noise=0.05): DataSet {
  const xs: number[][] = [], ys: number[] = [];
  for (let i=0;i<m;i++){
    const x = Math.random()*2-1, y = Math.random()*2-1;
    const lab = x*y > 0 ? 1 : 0;
    const [nx,ny] = addNoise([x,y], noise);
    xs.push([nx,ny]); ys.push(lab);
  }
  return { xs, ys };
}

export function genGaussian(m=400, noise=0.05): DataSet {
  const xs: number[][] = [], ys: number[] = [];
  const centers = [[-0.5,-0.5],[0.5,0.5]];
  for (let i=0;i<m;i++){
    const c = Math.random()<0.5 ? 0 : 1;
    const x = centers[c][0] + randn()*0.3;
    const y = centers[c][1] + randn()*0.3;
    const [nx,ny] = addNoise([x,y], noise);
    xs.push([nx,ny]); ys.push(c);
  }
  return { xs, ys };
}

export function genSpiral(m=400, noise=0.05): DataSet {
  const xs: number[][] = [], ys: number[] = [];
  const h = m/2;
  for (let i=0;i<h;i++){
    const r = i/h, t = 4.5*r*Math.PI;
    const [x1,y1] = addNoise([ r*Math.cos(t), r*Math.sin(t) ], noise);
    const [x2,y2] = addNoise([ r*Math.cos(t+Math.PI), r*Math.sin(t+Math.PI) ], noise);
    xs.push([x1,y1]); ys.push(1);
    xs.push([x2,y2]); ys.push(0);
  }
  return { xs, ys };
}

export function genLinear(m=400, noise=0.05): DataSet {
  const xs: number[][] = [], ys: number[] = [];
  const slope = 0.5, intercept = 0;
  for (let i=0;i<m;i++){
    const x = Math.random()*2-1, y = Math.random()*2-1;
    const lab = y > slope*x + intercept ? 1 : 0;
    const [nx,ny] = addNoise([x,y], noise);
    xs.push([nx,ny]); ys.push(lab);
  }
  return { xs, ys };
}

function randn(){
  let u=0, v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
  return Math.sqrt(-2.0*Math.log(u))*Math.cos(2*Math.PI*v);
}
