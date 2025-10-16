import * as tf from '@tensorflow/tfjs'
import type { Settings } from '../types'
import { makeFeatureFn, featureDim } from './model'

export interface DrawInputs {
  ctx: CanvasRenderingContext2D;
  width: number; height: number;
  model: tf.LayersModel | null;
  settings: Settings;
}

export function toPx(x:number, y:number, w:number, h:number){
  const px = (x+1)/2 * w;
  const py = (1-(y+1)/2) * h;
  return [px, py];
}

export function clear(ctx: CanvasRenderingContext2D, w: number, h: number){
  ctx.clearRect(0,0,w,h);
}

export async function drawDecision(
  { ctx, width:w, height:h, model, settings }: DrawInputs
){
  if (!model) return;
  const cell = settings.gridResolution;
  const cols = Math.ceil(w / cell);
  const rows = Math.ceil(h / cell);

  const feat = makeFeatureFn(settings.features);
  const dim = featureDim(settings.features);

  const points: number[] = [];
  for (let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      const x = (c*cell + cell/2)/w * 2 - 1;
      const y = 1 - (r*cell + cell/2)/h * 2;
      const f = feat(x,y);
      for (let k=0;k<dim;k++) points.push(f[k] ?? 0);
    }
  }

  const X = tf.tensor2d(points, [rows*cols, dim]);
  const P = model.predict(X) as tf.Tensor;
  const preds = await P.data();
  X.dispose(); P.dispose();

  let i=0;
  for (let r=0;r<rows;r++){
    for (let c=0;c<cols;c++){
      const p = preds[i++];
      const g = Math.min(255, Math.max(0, Math.round(p*255)));
      const rch = 255 - g;
      ctx.fillStyle = `rgba(${rch},${g},140,0.20)`;
      ctx.fillRect(c*cell, r*cell, cell, cell);
    }
  }
}

export function drawPoints(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  data: {xs:number[][], ys:number[]},
  splitIndex: number, showTest: boolean
){
  const rad = 3.5;
  ctx.lineWidth = 1;
  for (let i=0;i<splitIndex;i++){
    const [x,y] = data.xs[i]; const lab = data.ys[i];
    const [px,py] = toPx(x,y,w,h);
    ctx.fillStyle = lab ? '#9ef0a7' : '#ff9e9e';
    ctx.strokeStyle = '#0b0f20';
    ctx.beginPath(); ctx.arc(px,py,rad,0,Math.PI*2); ctx.fill(); ctx.stroke();
  }
  if (!showTest) return;
  ctx.globalAlpha = 0.65;
  for (let i=splitIndex;i<data.xs.length;i++){
    const [x,y] = data.xs[i]; const lab = data.ys[i];
    const [px,py] = toPx(x,y,w,h);
    ctx.fillStyle = lab ? '#9ef0a7' : '#ff9e9e';
    ctx.strokeStyle = '#0b0f20';
    ctx.beginPath(); ctx.rect(px-3,py-3,6,6); ctx.fill(); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}
